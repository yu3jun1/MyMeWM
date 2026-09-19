# Stage 1 补充实验方案：MRI-CORE + LoRA Representation Adaptation

## 1. 实验目的

当前 Stage 1 的主要结果显示：

- **BrainIAC latent space**：Recursive Rollout Training（RRT）能够稳定降低多步 rollout error。
- **Frozen MRI-CORE latent space**：RRT 未获得同样收益，Long MSE 反而略有上升。
- 因此目前能够支持的结论是：**RRT 的收益具有 representation dependence**，尚不能宣称对不同 latent representations 均有效。

本补充实验的目标不是证明 MRI-CORE 一定优于 BrainIAC，而是回答一个更具体的问题：

> **MRI-CORE 在经过面向 longitudinal patient dynamics 的 LoRA adaptation 后，RRT 是否能够重新获得多步预测收益？**

核心假设：

\[
\text{RRT effectiveness}
\quad \text{depends on} \quad
\text{dynamics compatibility of the latent representation}.
\]

因此，本实验将 **LoRA representation adaptation** 与 **RRT** 作为两个独立因素进行控制。

---

# 2. Research Questions

## RQ3：MRI-CORE 的 dynamics adaptation 是否改善 latent dynamics prediction？

比较：

\[
\text{Frozen MRI-CORE}
\quad \text{vs.} \quad
\text{LoRA-adapted MRI-CORE}.
\]

重点观察：

- MSE@1
- MSE@2
- MSE@3
- Long MSE

目的不是比较不同 latent space 的绝对 MSE，而是在 **MRI-CORE 内部**比较 adaptation 前后的变化。

---

## RQ4：LoRA adaptation 后，RRT 是否重新获得收益？

核心比较：

\[
\text{LoRA MRI-CORE + Baseline}
\quad \text{vs.} \quad
\text{LoRA MRI-CORE + RRT}.
\]

若：

\[
\operatorname{MSE}_{\text{RRT}}
<
\operatorname{MSE}_{\text{Baseline}},
\]

尤其在 H2/H3 上成立，则支持：

> RRT 的效果与 latent representation 是否适合 longitudinal dynamics modeling 有关。

---

# 3. 实验定位

本实验是 **Stage 1 的 representation adaptation experiment**，不是新的主任务。

仍然不加入：

- Policy model
- Treatment recommendation
- Survival / Outcome model
- Counterfactual treatment effect estimation
- Clinical decision evaluation

仍然只回答：

> 给定真实历史 treatment sequence 和时间间隔，能否更稳定地预测未来 latent patient state？

---

# 4. 与 CLARITY 的关系

CLARITY 的核心思路之一是使用可训练的 online vision encoder，并通过参数高效适配使视觉 representation 更适合 longitudinal disease modeling。

当前公开代码默认使用 BrainIAC online encoder，并提供 LoRA fine-tuning，同时 vision backbone 接口是可替换的。

本实验不要求完全复现 CLARITY 的完整训练目标，而是借鉴其核心思想：

\[
\boxed{
\text{Pretrained MRI Encoder}
+
\text{Task-specific LoRA Adaptation}
}
\]

并专门研究这种 adaptation 是否会影响 RRT 的有效性。

MRI-CORE 官方实现基于 SAM ViT-B，因此 LoRA 可以优先插入 Transformer attention 中的 Q/V projection。

---

# 5. 核心实验矩阵

必须保留四个主要实验组。

| Group | Encoder | Dynamics Training | 目的 |
|---|---|---|---|
| A | Frozen MRI-CORE | One-step Baseline | 已有 MRI-CORE baseline |
| B | Frozen MRI-CORE | RRT, \(K_{\max}=3\) | 已有 MRI-CORE RRT |
| C | MRI-CORE + LoRA | One-step Baseline | 测试 LoRA adaptation 本身 |
| D | MRI-CORE + LoRA | RRT, \(K_{\max}=3\) | 测试 LoRA 后 RRT 是否恢复收益 |

最重要的比较不是：

\[
A \rightarrow D
\]

而是：

\[
\boxed{C \rightarrow D}
\]

因为只有 C 和 D 使用相同的 LoRA 设置。

这样才能回答：

> 在同样经过 adaptation 的 MRI-CORE representation 下，RRT 是否优于 one-step training？

---

# 6. 推荐实验流程

建议将实验拆成两个阶段。

---

## Phase A：MRI-CORE LoRA Adaptation

目标：

> 将 frozen MRI-CORE 从通用 MRI representation 轻量适配到 longitudinal treatment-conditioned dynamics representation。

### 6.1 输入

对于每个真实 transition：

\[
(x_t, A_t, \Delta t_t, x_{t+1}),
\]

其中：

- \(x_t\)：当前真实 MRI
- \(x_{t+1}\)：下一时间点真实 MRI
- \(A_t\)：真实历史 treatment
- \(\Delta t_t\)：真实时间间隔

MRI preprocessing 必须与当前 MRI-CORE feature extraction pipeline 保持一致。

不要在该实验中同时改变：

- MRI slice selection
- modality composition
- spatial preprocessing
- latent pooling
- normalization

否则无法判断变化来自 LoRA 还是 preprocessing。

---

## 6.2 Encoder

初始化：

\[
E_{\theta}
=
\text{Pretrained MRI-CORE}.
\]

冻结原始参数：

\[
\theta_{\text{base}}
\quad \text{frozen}.
\]

仅训练 LoRA：

\[
\theta'
=
\theta_{\text{base}}+\Delta\theta_{\text{LoRA}}.
\]

推荐默认设置：

```yaml
encoder: mri_core
encoder_train_mode: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target: attention_qv
```

由于 MRI-CORE 基于 SAM ViT-B，优先对 Transformer attention 的 Q/V 分支增加 LoRA。

如果代码中使用 fused `qkv` Linear：

```text
attn.qkv
```

则实现对其中 Q/V 子空间的 LoRA update，而不是重新训练完整 qkv 权重。

第一轮实验不建议同时对：

- MLP
- patch embedding
- layer norm
- output projection

全部加入 LoRA。

先保持变量最少。

---

# 7. LoRA Adaptation Objective

建议使用一个独立的 one-step dynamics adapter 进行 LoRA adaptation。

定义：

\[
z_t^{L}=E_{\theta+\Delta\theta}(x_t),
\]

\[
z_{t+1}^{L}=E_{\theta+\Delta\theta}(x_{t+1}).
\]

one-step transition model：

\[
\hat z_{t+1}
=
f_{\phi}
(
z_t^{L},
A_t,
\Delta t_t
).
\]

主要 dynamics loss：

\[
\mathcal L_{\text{dyn}}
=
\left\|
\hat z_{t+1}
-
\operatorname{sg}(z_{t+1}^{L})
\right\|_2^2.
\]

其中 `sg` 表示 stop-gradient。

---

## 7.1 Representation Anchor

由于数据量较小，不建议让 MRI-CORE latent space 完全自由漂移。

保留 frozen MRI-CORE：

\[
z_t^{0}=E_{\theta}(x_t).
\]

增加 representation anchor：

\[
\mathcal L_{\text{anchor}}
=
1-
\cos
(
z_t^{L},
z_t^{0}
).
\]

最终：

\[
\mathcal L_{\text{adapt}}
=
\mathcal L_{\text{dyn}}
+
\beta
\mathcal L_{\text{anchor}}.
\]

推荐第一版：

```yaml
anchor_weight: 0.1
```

可选 sensitivity：

```text
β ∈ {0.05, 0.1, 0.2}
```

但不要把该 sweep 放入主实验。

---

# 8. 为什么 Phase A 只使用 one-step objective

LoRA adaptation 阶段不要直接使用 RRT。

原因：

如果：

- LoRA representation 使用 RRT 学习
- 后续又比较 Baseline vs RRT

那么 RRT 已经参与 encoder adaptation，会产生方法泄漏。

因此推荐：

\[
\boxed{
\text{LoRA adaptation uses only one-step transitions}
}
\]

然后：

1. 保存 LoRA-adapted encoder；
2. 冻结 encoder；
3. 重新提取所有 patient timepoint latent；
4. 再进行 Baseline vs RRT 的公平比较。

这样 LoRA representation 对两个 dynamics training methods 完全一致。

---

# 9. 防止数据泄漏

LoRA adaptation 必须严格使用：

\[
\boxed{\text{train patients only}}
\]

不能用：

- validation MRI 更新 LoRA
- test MRI 更新 LoRA
- 全数据集 jointly adapt encoder

patient split 必须与当前 Stage 1 完全一致。

例如主 split：

```yaml
split_seed: 17
train_fraction: 0.70
validation_fraction: 0.15
test_fraction: 0.15
```

流程：

```text
Raw MRI
   │
   ├── Train patients
   │      └── LoRA adaptation
   │
   ├── Validation patients
   │      └── inference only
   │
   └── Test patients
          └── inference only
```

---

# 10. Adapted Latent Extraction

完成 LoRA adaptation 后：

```text
MRI-CORE pretrained weights
        +
trained LoRA weights
        ↓
freeze encoder
        ↓
extract latent for every MRI timepoint
        ↓
MRI-CORE-LoRA latent dataset
```

建议保存为独立 representation：

```text
data/latent/
├── brainiac/
├── mri_core_frozen/
└── mri_core_lora/
```

不要覆盖原始 MRI-CORE features。

---

# 11. Latent Normalization

MRI-CORE-LoRA 需要重新计算 normalization statistics。

仅使用：

\[
\text{training patients}
\]

计算：

\[
\mu_{\text{train}},
\qquad
\sigma_{\text{train}}.
\]

然后：

\[
\tilde z
=
\frac{z-\mu_{\text{train}}}
{\sigma_{\text{train}}+\epsilon}.
\]

Validation/Test 必须使用同一组 train statistics。

不要复用 frozen MRI-CORE 的 normalization statistics。

---

# 12. Phase B：重新运行 Stage 1 Dynamics Experiment

得到 frozen MRI-CORE-LoRA features 后，后续代码应完全复用现有 Stage 1 pipeline。

---

## 12.1 Group C：LoRA MRI-CORE + Baseline

训练方式：

```yaml
variant: baseline
training_horizon_strategy: one_step
max_horizon: 1
terminal_loss_only: true
teacher_forcing: false
```

训练：

\[
z_t
\xrightarrow{A_t,\Delta t_t}
\hat z_{t+1}.
\]

---

## 12.2 Group D：LoRA MRI-CORE + RRT

保持当前正式 RRT：

```yaml
variant: rrt
training_horizon_strategy: max_available
main_max_horizon: 3
terminal_loss_only: true
teacher_forcing: false
```

例如 \(k=3\)：

\[
z_t
\rightarrow
\hat z_{t+1}
\rightarrow
\hat z_{t+2}
\rightarrow
\hat z_{t+3}.
\]

中间状态全部使用模型预测：

\[
\hat z_{t+j}.
\]

Action 和 delta time 使用真实历史序列：

\[
A_{t:t+k-1},
\quad
\Delta t_{t:t+k-1}.
\]

只对 terminal state 计算 loss：

\[
\mathcal L_{\mathrm{RRT}}
=
\|
\hat z_{t+k}
-
z_{t+k}
\|_2^2.
\]

---

# 13. Dynamics Model 必须保持完全一致

LoRA experiment 不改变：

```yaml
hidden_dim: 128
action_embed_dim: 32
time_embed_dim: 16
batch_size: 32
epochs: 100
learning_rate: 0.001
weight_decay: 0.0001
gradient_clip_norm: 1.0
early_stopping_patience: 15
checkpoint_metric: val_recursive_mse_k2_k3
```

目的：

> 唯一新增变量是 MRI-CORE representation 是否经过 LoRA adaptation。

---

# 14. LoRA Adaptation 推荐训练参数

由于 online MRI encoder 显存开销较大，LoRA adaptation 单独使用一套 config。

推荐初始设置：

```yaml
mri_core_lora:
  rank: 8
  alpha: 16
  dropout: 0.05

  target_modules:
    - attention_q
    - attention_v

  encoder_lr: 0.0001
  dynamics_lr: 0.001

  weight_decay: 0.0001
  grad_clip_norm: 1.0

  epochs: 50
  early_stopping_patience: 10

  anchor_weight: 0.1

  physical_batch_size: 4
  gradient_accumulation_steps: 8
  effective_batch_size: 32
```

如果显存允许，可提高 physical batch size。

第一轮实验不要搜索大量 LoRA 超参数。

---

# 15. Checkpoint Selection

LoRA adaptation checkpoint 不使用 test set。

推荐使用 validation one-step dynamics MSE：

\[
\operatorname{ValMSE@1}.
\]

即：

```yaml
lora_checkpoint_metric: val_one_step_mse
```

选择最低 validation MSE 对应的 LoRA checkpoint。

然后固定该 encoder。

---

# 16. Seeds

## LoRA Adaptation

为了控制计算成本，第一阶段先固定：

```text
LoRA seed = 17
```

适配得到一个共享的 MRI-CORE-LoRA representation。

---

## Dynamics Training

继续沿用当前 Stage 1：

```text
training seeds = [7, 17, 29]
```

因此 Group C 和 Group D 都运行：

```text
seed 7
seed 17
seed 29
```

这样 RRT 相对改善仍然可以做 paired comparison。

如果主结果成立，再考虑增加 LoRA seed robustness。

---

# 17. 主评估指标

与现有 Stage 1 完全一致。

### Horizon-specific error

\[
\operatorname{MSE@1},
\quad
\operatorname{MSE@2},
\quad
\operatorname{MSE@3}.
\]

### Long-horizon MSE

\[
\operatorname{LongMSE}
=
\frac{
\operatorname{MSE@2}
+
\operatorname{MSE@3}
}{2}.
\]

### Relative Improvement

对于 LoRA MRI-CORE：

\[
RI_h
=
\frac{
MSE_h^{\text{LoRA-Baseline}}
-
MSE_h^{\text{LoRA-RRT}}
}{
MSE_h^{\text{LoRA-Baseline}}
}
\times100\%.
\]

重点：

\[
RI_{\text{Long}}.
\]

---

# 18. 主结果表

建议新增独立表：

## Table: Effect of LoRA Adaptation on RRT

| Representation | Training | MSE@1 ↓ | MSE@2 ↓ | MSE@3 ↓ | Long MSE ↓ | RI Long |
|---|---|---:|---:|---:|---:|---:|
| MRI-CORE Frozen | Baseline | existing | existing | existing | existing | — |
| MRI-CORE Frozen | RRT | existing | existing | existing | existing | existing |
| MRI-CORE LoRA | Baseline | new | new | new | new | — |
| MRI-CORE LoRA | RRT | new | new | new | new | **new** |

报告：

```text
mean ± std over training seeds [7,17,29]
```

---

# 19. 最重要的结果解释

实验结果可能出现四种情况。

---

## Case 1：LoRA 后 RRT 明显有效

例如：

\[
RI_{\text{Long}}>0
\]

且三个 seed 都为正。

这是最有价值的结果。

可以支持：

> RRT effectiveness depends on the compatibility between the latent representation and longitudinal patient dynamics. After task-specific MRI-CORE adaptation, recursive rollout training again improves multi-step prediction.

研究故事变成：

\[
\boxed{
\text{Dynamics-compatible Representation}
+
\text{RRT}
\rightarrow
\text{Better Long-horizon Rollout}
}
\]

---

## Case 2：LoRA 本身改善 Baseline，但 RRT 仍无额外收益

例如：

\[
A > C
\]

但：

\[
C \approx D.
\]

说明：

> MRI-CORE representation 可以通过 adaptation 改善 dynamics prediction，但 RRT 的收益仍不是 representation-independent。

此时不要声称 LoRA 恢复了 RRT。

RRT contribution 仍主要建立在 BrainIAC 上。

---

## Case 3：LoRA 后 Baseline 和 RRT 都改善，RRT 仍优于 Baseline

这是最理想情况：

\[
C < A,
\]

同时：

\[
D < C.
\]

可以将两个效应区分为：

1. LoRA improves representation dynamics compatibility.
2. RRT further reduces recursive rollout mismatch.

---

## Case 4：LoRA 后整体反而更差

说明当前数据量不足以稳定 adaptation，或 adaptation objective 与真实 patient dynamics 不匹配。

不要继续扩大 LoRA 超参数搜索来追求正结果。

应报告：

> With the limited longitudinal cohort, adapting MRI-CORE does not reliably improve recursive dynamics modeling.

然后保持 RRT contribution 的 representation-specific 定位。

---

# 20. 不建议直接比较 BrainIAC 与 MRI-CORE 的绝对 MSE

仍然保持原实验原则：

\[
MSE_{\text{BrainIAC}}
\not\leftrightarrow
MSE_{\text{MRI-CORE}}
\]

因为：

- latent geometry 不同
- feature scale 不同
- representation distribution 不同

正确比较：

### BrainIAC 内部

\[
Baseline
\rightarrow
RRT
\]

### Frozen MRI-CORE 内部

\[
Baseline
\rightarrow
RRT
\]

### LoRA MRI-CORE 内部

\[
Baseline
\rightarrow
RRT.
\]

---

# 21. 推荐增加一个 representation diagnostic

为了验证 LoRA 确实改变了 latent representation，而不是训练失败，建议增加两个简单指标。

---

## 21.1 Representation Drift

对同一个 MRI：

\[
D_{\text{LoRA}}
=
1-
\cos
(
z^{0},
z^{L}
).
\]

报告 train / validation / test 的平均值。

用途：

> 确认 LoRA 是否产生了非零但不过度的 representation change。

不作为主性能指标。

---

## 21.2 Transition Smoothness

对于真实连续 states：

\[
S
=
\frac{1}{N}
\sum_t
\left(
1-
\cos(z_t,z_{t+1})
\right).
\]

分别报告：

```text
Frozen MRI-CORE
LoRA MRI-CORE
```

仅作为 descriptive analysis。

不要预设：

```text
smaller = always better
```

因为疾病状态本身可能发生显著变化。

---

# 22. 可选：Patient-level Macro MSE

当前 H3 test window 数量较少，且同一 patient 可能贡献多个 windows。

建议在补充实验中额外计算：

1. 每个 patient 内先平均 prediction error；
2. 再对 patient 求平均。

定义：

\[
MSE^{patient}
=
\frac{1}{P}
\sum_{p=1}^{P}
\left[
\frac{1}{N_p}
\sum_{i\in p}
e_i
\right].
\]

这可以避免拥有更多 longitudinal windows 的 patient 对整体结果产生过大权重。

作为 robustness metric 即可，不需要重新引入 95% CI。

---

# 23. Ensemble 暂时不要同时加入主实验

第一轮只验证：

```text
LoRA-Baseline
vs.
LoRA-RRT
```

原因：

同时加入：

- LoRA
- RRT
- Ensemble

会使结果难以解释。

只有当：

\[
\boxed{
\text{LoRA MRI-CORE + RRT}
<
\text{LoRA MRI-CORE + Baseline}
}
\]

成立后，再补：

```text
LoRA MRI-CORE + Ensemble
LoRA MRI-CORE + RRT + Ensemble
```

用于测试 trajectory-level disagreement 是否同样有效。

---

# 24. 推荐代码结构

在当前仓库中增加：

```text
src/clarity_hauwm/
├── encoders/
│   ├── __init__.py
│   ├── mri_core.py
│   └── mri_core_lora.py
│
├── lora_adaptation.py
├── extract_adapted_latents.py
├── training.py
├── evaluation.py
└── reporting.py
```

---

## 24.1 `mri_core_lora.py`

职责：

```text
1. load pretrained MRI-CORE
2. freeze base parameters
3. inject LoRA into attention Q/V
4. expose forward()
5. return same latent format as existing MRI-CORE pipeline
```

接口建议：

```python
class MRICoreLoRAEncoder(nn.Module):

    def __init__(
        self,
        checkpoint,
        rank=8,
        alpha=16,
        dropout=0.05,
    ):
        ...

    def forward(self, mri):
        ...
        return latent
```

---

## 24.2 `lora_adaptation.py`

负责：

```text
train-patient-only LoRA adaptation
validation checkpoint selection
representation anchor
checkpoint saving
```

输出：

```text
outputs/stage1/mri_core_lora/adaptation/
├── config.json
├── best_lora.pt
├── training_log.json
└── adaptation_summary.md
```

---

## 24.3 `extract_adapted_latents.py`

输入：

```text
raw MRI
+
pretrained MRI-CORE
+
best_lora.pt
```

输出：

```text
outputs/stage1/mri_core_lora/features/
├── train.pt
├── validation.pt
├── test.pt
├── normalization.json
└── provenance.json
```

`provenance.json` 必须记录：

```json
{
  "base_encoder": "MRI-CORE",
  "adaptation": "LoRA",
  "lora_rank": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "split_seed": 17,
  "adaptation_seed": 17,
  "train_patients_only": true
}
```

---

# 25. Config 增加

建议增加：

```json
{
  "mri_core_lora": {
    "enabled": true,
    "rank": 8,
    "alpha": 16,
    "dropout": 0.05,
    "target_modules": ["q", "v"],
    "encoder_lr": 0.0001,
    "adaptation_dynamics_lr": 0.001,
    "anchor_weight": 0.1,
    "epochs": 50,
    "early_stopping_patience": 10,
    "adaptation_seed": 17,
    "checkpoint_metric": "val_one_step_mse"
  }
}
```

---

# 26. 输出目录

建议不要把 LoRA 结果混入当前 MRI-CORE 目录。

```text
outputs/stage1/
├── brainiac/
├── mri_core/
│   └── frozen/
└── mri_core_lora/
    ├── adaptation/
    ├── features/
    ├── baseline/
    │   ├── seed_7/
    │   ├── seed_17/
    │   └── seed_29/
    ├── rrt/
    │   ├── seed_7/
    │   ├── seed_17/
    │   └── seed_29/
    └── reports/
        ├── prediction_metrics.json
        ├── paired_relative_improvement.json
        ├── patient_macro_metrics.json
        ├── representation_diagnostics.json
        └── summary.md
```

---

# 27. 执行顺序

## Step 1：实现 online MRI-CORE encoder

确保：

```text
raw MRI
→ MRI-CORE
→ latent
```

输出维度、pooling 和当前 frozen MRI-CORE feature extraction 保持一致。

---

## Step 2：加入 LoRA

冻结 base encoder。

验证：

```text
trainable parameters
≈ LoRA parameters only
```

打印：

```text
total parameters
trainable parameters
trainable ratio
```

---

## Step 3：LoRA adaptation

只使用 train patients。

训练：

```text
one-step dynamics objective
+
representation anchor
```

validation 选择 checkpoint。

---

## Step 4：冻结 adapted encoder

不再更新：

```text
MRI-CORE base
LoRA parameters
```

---

## Step 5：重新提取 latent

分别生成：

```text
train
validation
test
```

features。

---

## Step 6：重新计算 normalization

只使用 train adapted latent。

---

## Step 7：运行 LoRA-Baseline

```text
training seeds = 7, 17, 29
```

---

## Step 8：运行 LoRA-RRT

```text
Kmax = 3
training seeds = 7, 17, 29
```

---

## Step 9：统一评价 H1/H2/H3

输出：

```text
MSE@1
MSE@2
MSE@3
Long MSE
paired RI
patient-level macro MSE
```

---

## Step 10：根据结果决定是否继续 Ensemble

只有 LoRA-RRT 主结论成立后，再加入：

```text
LoRA-ensemble
LoRA-RRT-ensemble
```

---

# 28. 最小可执行实验集

如果希望尽量控制计算成本，最低只需要新增：

### Encoder adaptation

```text
MRI-CORE + LoRA
seed = 17
```

### Dynamics

```text
LoRA MRI-CORE + Baseline
    seed 7
    seed 17
    seed 29

LoRA MRI-CORE + RRT
    seed 7
    seed 17
    seed 29
```

共：

```text
1 次 LoRA adaptation
+
6 次 dynamics training
```

即可回答核心问题。

---

# 29. 判断标准

不设置自动 pass / fail threshold。

重点检查：

### 条件 1

\[
MSE@2_{\text{RRT}}
<
MSE@2_{\text{Baseline}}
\]

### 条件 2

\[
MSE@3_{\text{RRT}}
<
MSE@3_{\text{Baseline}}
\]

### 条件 3

\[
LongMSE_{\text{RRT}}
<
LongMSE_{\text{Baseline}}
\]

### 条件 4

三个 training seeds 的：

\[
RI_{\text{Long}}
\]

方向是否一致。

不要求人为规定：

```text
RI > 5%
```

才算成功。

---

# 30. 论文中的推荐定位

如果 LoRA 后 RRT 恢复明显收益，可以将结果解释为：

> The effectiveness of recursive rollout training depends on the compatibility of the latent representation with longitudinal disease dynamics. While RRT provides limited gains with frozen MRI-CORE features, task-specific LoRA adaptation enables the representation to better support recursive state transitions, after which RRT again improves multi-step prediction.

贡献逻辑：

\[
\boxed{
\text{Representation Adaptation}
\rightarrow
\text{Dynamics-compatible Latent Space}
\rightarrow
\text{RRT}
\rightarrow
\text{Reduced Multi-step Error}
}
\]

但不要写成：

> LoRA proves that MRI-CORE was previously unsuitable.

除非有更强的 representation analysis。

---

# 31. 如果 LoRA 后仍无效

如果：

\[
RI_{\text{Long}}\le0,
\]

则不继续增加大量 tuning。

论文中保留结论：

> RRT substantially improves multi-step dynamics prediction in the BrainIAC representation, whereas the same benefit is not consistently observed with MRI-CORE, even after lightweight task-specific adaptation.

此时将结果解释为：

\[
\boxed{
\text{RRT is representation-dependent}
}
\]

并将 “representation suitability for recursive patient dynamics” 作为 limitation / future work。

这样的结论仍然比隐藏 MRI-CORE negative result 更稳妥。

---

# 32. 与现有 Stage 1 的最终关系

最终 Stage 1 可以形成三层证据：

### Evidence 1：RRT effectiveness

BrainIAC：

\[
K1 \rightarrow K2 \rightarrow K3
\]

证明增加 recursive rollout exposure 可以降低 multi-step error。

### Evidence 2：Representation dependence

Frozen MRI-CORE：

\[
\text{RRT gain is weak / absent}.
\]

说明 RRT 并非自动适用于任意 latent representation。

### Evidence 3：Representation adaptation

MRI-CORE + LoRA：

验证 task-specific representation adaptation 是否可以恢复 RRT effectiveness。

最终可以形成更完整的研究问题：

\[
\boxed{
\text{How do representation learning and recursive training jointly affect
long-horizon patient dynamics modeling?}
}
\]

这比单独证明 “RRT 在两个 encoder 上都有效” 更有研究解释力。

---

# 33. 当前推荐优先级

按优先级执行：

1. **实现 MRI-CORE online LoRA encoder**
2. **完成 train-patient-only one-step LoRA adaptation**
3. **冻结并重新提取 MRI-CORE-LoRA latent**
4. **运行 LoRA-Baseline**
5. **运行 LoRA-RRT**
6. **比较 H1/H2/H3 与 Long MSE**
7. **补 patient-level macro error**
8. 只有主结果成立后再加入 Ensemble

当前不建议优先做：

- LoRA rank 大规模 sweep
- Kmax=5
- Ensemble + LoRA 全组合
- Survival / Policy
- 第三个 encoder

先把最关键的因果关系验证清楚：

\[
\boxed{
\text{Does dynamics-oriented LoRA adaptation restore the benefit of RRT on MRI-CORE?}
}
\]

---

# 参考资料

1. CLARITY: Medical World Model for Guiding Treatment Decisions by Modeling Context-Aware Disease Trajectories in Latent Space  
   https://arxiv.org/abs/2512.08029

2. CLARITY official repository  
   https://github.com/DingTianxingjian/CLARITY

3. MRI-CORE official repository  
   https://github.com/mazurowski-lab/mri_foundation

> 注：截至本方案整理时，CLARITY 当前公开代码默认展示 BrainIAC online encoder + LoRA，并提供可插拔 vision backbone 接口；本文档借鉴的是其“online encoder task adaptation”思路，而不是声称当前公开代码已经完整提供 MRI-CORE + LoRA 的现成训练脚本。
