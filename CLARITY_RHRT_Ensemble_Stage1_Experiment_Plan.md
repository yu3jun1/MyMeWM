# CLARITY × Random-Horizon Recursive Training × Ensemble
## Stage 1：最小迁移验证实验方案

## 0. 本次修改点

相较于当前 Stage 1，实现和实验设计做以下调整。

### 0.1 将 Horizon Sampling 改为 Random-Horizon Recursive Training

当前实现：

\[
(z_t,A_{t:t+k-1})\rightarrow \hat z_{t+k}
\]

即一次性输入完整真实 treatment sequence，直接预测远期 latent。

修改为：

\[
\hat z_{t+1}=f_\theta(z_t,A_t,\Delta t_t)
\]

\[
\hat z_{t+2}=f_\theta(\hat z_{t+1},A_{t+1},\Delta t_{t+1})
\]

\[
\cdots
\]

\[
\hat z_{t+k}=f_\theta(\hat z_{t+k-1},A_{t+k-1},\Delta t_{t+k-1})
\]

训练时随机采样：

\[
k\sim\{1,2,3\}
\]

只有起始状态 \(z_t\) 使用真实 latent。

从第二步开始，必须使用模型自己的预测结果作为下一步输入。

默认只在最终 horizon \(t+k\) 进行监督：

\[
\mathcal L_{\text{RHRT}}
=
D(\hat z_{t+k},z_{t+k})
\]

中间步骤不使用真实 latent 校正，不进行 teacher forcing。

### 0.2 最大 Horizon 从 5 调整为 3

正式实验：

\[
K_{\max}=3
\]

原因是 MU-Glioma-Post 的长轨迹患者数量快速下降。主要实验不再依赖 \(k=4,5\)。

正式结论只考虑：

\[
k=1,2,3
\]

这样可以提高训练样本覆盖和统计稳定性。

### 0.3 主要研究问题从 Direct Prediction 改为 Recursive Rollout

旧方案主要关注：

\[
z_t\rightarrow z_{t+k}
\]

的 direct prediction。

新方案主要关注：

\[
z_t
\rightarrow
\hat z_{t+1}
\rightarrow
\hat z_{t+2}
\rightarrow
\hat z_{t+3}
\]

因此 primary metrics 全部围绕 recursive rollout。

Direct prediction 不再作为主要实验。

### 0.4 Ensemble 的定位明确为 Reliability Estimation

Ensemble 不以提高预测精度作为主要目标。

其主要作用为：

\[
\text{ensemble disagreement}
\rightarrow
\text{prediction uncertainty}
\]

验证：

\[
U\uparrow
\quad\Rightarrow\quad
Prediction\ Error\uparrow
\]

即 uncertainty 能否识别不可靠的 imagined trajectory。

### 0.5 指标输出拆分

取消单个大型 `encoder_comparison.json` 汇总所有信息的方式。

训练、recursive prediction、uncertainty、bootstrap test 和最终结论分别保存。

多指标结果在终端和日志中使用表格打印。

不保存 CSV。

JSON 用于后续程序读取。

---

## 1. 实验目标

Stage 1 只回答两个核心研究问题。

### RQ1

**Random-Horizon Recursive Training 是否能够减少多步 patient-state rollout 中的误差累积？**

比较：

\[
\text{One-Step Training}
\]

和：

\[
\text{Random-Horizon Recursive Training}
\]

重点观察：

\[
k=2,3
\]

时 recursive prediction error 是否下降。

### RQ2

**Independent Dynamics Ensemble 的 disagreement 是否能够识别高风险预测？**

验证：

\[
\text{Ensemble Disagreement}
\]

是否与：

\[
\|\hat z-z\|^2
\]

正相关。

Stage 1 不研究：

- Treatment recommendation
- Policy optimization
- Survival prediction
- Causal treatment effect
- Counterfactual treatment ranking

---

## 2. 数据与 Encoder

### 2.1 数据集

使用：

`MU-Glioma-Post`

使用已有临床 timeline、MRI timepoint 和真实 treatment sequence。

每个 transition 表示：

\[
(z_t,A_t,\Delta t_t,z_{t+1})
\]

其中：

- \(z_t\)：当前 MRI latent
- \(A_t\)：当前 MRI 到下一次 MRI 之间的治疗
- \(\Delta t_t\)：时间间隔
- \(z_{t+1}\)：下一次真实 MRI latent

继续采用：

`action_anchor = source`

避免使用目标 timepoint 才出现的信息。

### 2.2 Encoder

主实验建议使用：

**BrainIAC**

Encoder 全程冻结。

\[
MRI_t\rightarrow z_t\in\mathbb R^{768}
\]

MRI-CORE 使用完全相同实验协议作为 robustness experiment。

两个 encoder：

- 不联合训练
- 不共享 dynamics
- 不直接比较绝对 latent MSE

只比较：

\[
\text{method vs baseline}
\]

在各自 latent space 内的相对改善。

---

## 3. 数据划分

继续采用 patient-level split：

| Split | Ratio |
|---|---:|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

固定：

`split_seed = 17`

同一个患者的所有 timepoint 必须属于同一 split。

所有 variant 使用完全相同：

- Patients
- MRI latent
- Treatment
- Delta days
- Evaluation windows

---

## 4. Training Window 构建

对于患者：

\[
z_1,z_2,\ldots,z_T
\]

从任意合法起点 \(t\) 构建训练 window。

该位置最大可使用 horizon：

\[
K_t=\min(3,T-t)
\]

例如：

```text
Patient A

z1 → z2 → z3 → z4 → z5
```

从 \(z_1\)：

\[
K_1=3
\]

可采样：

\[
k\in\{1,2,3\}
\]

从 \(z_3\)：

\[
K_3=2
\]

只能采样：

\[
k\in\{1,2\}
\]

因此：

\[
k\sim Uniform(1,K_t)
\]

不得先采样 \(k\)，再因为轨迹长度不足丢弃样本。

---

## 5. 四组实验

| Variant | Recursive Horizon Training | Ensemble |
|---|---:|---:|
| `baseline` | No | No |
| `rhrt` | Yes | No |
| `ensemble` | No | Yes |
| `rhrt_ensemble` | Yes | Yes |

其中 RHRT 表示：

**Random-Horizon Recursive Training**

正式代码中不再使用容易产生歧义的 `hs` 名称。

---

## 6. Baseline Training

Baseline 保持普通 one-step dynamics training。

训练样本：

\[
(z_t,A_t,\Delta t_t,z_{t+1})
\]

预测：

\[
\hat z_{t+1}
=
f_\theta(z_t,A_t,\Delta t_t)
\]

Loss：

\[
\mathcal L_{\text{baseline}}
=
D(\hat z_{t+1},z_{t+1})
\]

Baseline 训练阶段永远使用真实 \(z_t\)。

---

## 7. Random-Horizon Recursive Training

### 7.1 Horizon Sampling

对于当前训练 window：

\[
k\sim Uniform(1,K_t)
\]

其中：

\[
K_t\le3
\]

### 7.2 Recursive Prediction

初始化：

\[
\hat z_t=z_t
\]

然后：

\[
\hat z_{t+i+1}
=
f_\theta(
\hat z_{t+i},
A_{t+i},
\Delta t_{t+i}
)
\]

其中：

\[
i=0,\ldots,k-1
\]

例如采样：

\[
k=3
\]

则：

```text
真实 z_t
   │
   │ A_t
   ▼
Dynamics
   │
   ▼
预测 ẑ_t+1
   │
   │ A_t+1
   ▼
Dynamics
   │
   ▼
预测 ẑ_t+2
   │
   │ A_t+2
   ▼
Dynamics
   │
   ▼
预测 ẑ_t+3
   │
   ▼
与真实 z_t+3 比较
```

注意：

\[
z_{t+1}
\]

和：

\[
z_{t+2}
\]

不重新输入模型。

### 7.3 Training Loss

主实验只使用 terminal loss：

\[
\mathcal L_{\text{RHRT}}
=
\frac{1}{d}
\|
\hat z_{t+k}-z_{t+k}
\|_2^2
\]

如果当前 latent 已标准化，则使用 normalized latent MSE。

暂时不加入：

\[
\mathcal L_{t+1}
+
\mathcal L_{t+2}
+
\mathcal L_{t+3}
\]

避免同时改变多个因素。

Intermediate supervision 可以留作后续 ablation。

---

## 8. Ensemble Dynamics

设置：

\[
M=5
\]

训练：

\[
f_{\theta_1},\ldots,f_{\theta_5}
\]

每个 member：

- 独立参数初始化
- 独立训练 random seed
- 独立 mini-batch 顺序
- 独立 horizon sampling

第一阶段不做 patient bootstrap，以避免进一步减少小数据集的有效覆盖。

---

## 9. Ensemble Recursive Rollout

Ensemble member 必须维护自己的 latent trajectory。

例如：

\[
\hat z_{t+1}^{(m)}
=
f_{\theta_m}(z_t,A_t)
\]

然后：

\[
\hat z_{t+2}^{(m)}
=
f_{\theta_m}(
\hat z_{t+1}^{(m)},A_{t+1}
)
\]

禁止先对：

\[
\hat z_{t+1}^{(1)},\ldots,\hat z_{t+1}^{(M)}
\]

求均值，再把 mean latent 输入所有 member。

否则 ensemble disagreement 会被人为压缩。

---

## 10. Ensemble Point Prediction

最终 ensemble prediction：

\[
\bar z_{t+k}
=
\frac1M
\sum_{m=1}^{M}
\hat z_{t+k}^{(m)}
\]

使用：

\[
\bar z
\]

计算 prediction MSE。

---

## 11. Ensemble Uncertainty

定义 disagreement：

\[
U_{t+k}
=
\frac1M
\sum_{m=1}^{M}
\frac1d
\|
\hat z_{t+k}^{(m)}
-
\bar z_{t+k}
\|_2^2
\]

它表示 epistemic disagreement。

真实 prediction error：

\[
E_{t+k}
=
\frac1d
\|
\bar z_{t+k}
-
z_{t+k}
\|_2^2
\]

然后研究：

\[
U
\quad vs\quad
E
\]

---

## 12. RQ1：Recursive Prediction Metrics

### 12.1 Primary Metric 1：Recursive MSE@k

分别计算：

\[
MSE@1
\]

\[
MSE@2
\]

\[
MSE@3
\]

全部从真实起始：

\[
z_t
\]

开始 recursive rollout。

越低越好。

终端打印：

```text
Recursive Rollout Performance

| Variant         | MSE@1 | MSE@2 | MSE@3 |
|-----------------|------:|------:|------:|
| baseline        |       |       |       |
| rhrt            |       |       |       |
| ensemble        |       |       |       |
| rhrt_ensemble   |       |       |       |
```

### 12.2 Primary Metric 2：Long-Horizon Recursive MSE

因为真正关注的是 error accumulation，因此定义：

\[
MSE_{\text{long}}
=
\frac{MSE@2+MSE@3}{2}
\]

主要比较：

\[
baseline
\]

与：

\[
rhrt
\]

以及：

\[
ensemble
\]

与：

\[
rhrt\_ensemble
\]

### 12.3 Primary Metric 3：Relative Improvement

对于 horizon \(k\)：

\[
RI_k=
\frac{
MSE_k^{base}
-
MSE_k^{RHRT}
}{
MSE_k^{base}
}
\]

例如：

\[
RI_3=0.10
\]

表示：

> RHRT 在三步 recursive rollout 上降低约 10% prediction error。

---

## 13. Error Growth Metric

为了验证 error accumulation，再计算：

\[
MSE(k)=\alpha+\beta k
\]

其中：

\[
k=1,2,3
\]

\[
\beta
=
\text{recursive error growth slope}
\]

越小越好。

但是不能直接对不同患者组成的 aggregate MSE 拟合 slope。

必须使用 **matched H3 cohort**：

只选择具有：

\[
k=1,2,3
\]

完整测试轨迹的患者。

对于每个患者单独计算：

\[
\beta_i
\]

然后再计算 patient-level mean slope。

这样：

\[
k=1,2,3
\]

来自同一批患者，可以避免 patient composition confounding。

---

## 14. Secondary Prediction Metric

同时报告：

\[
CosineDistance
=
1-
\frac{
\hat z^\top z
}{
\|\hat z\|\|z\|
}
\]

分别报告：

\[
Cos@1,\quad Cos@2,\quad Cos@3
\]

但它只作为 secondary metric。

正式结论仍以 normalized latent MSE 为主。

---

## 15. RQ2：Uncertainty Metrics

### 15.1 Primary Metric：Within-Horizon Spearman

分别计算：

\[
\rho_1
=
Spearman(U_1,E_1)
\]

\[
\rho_2
=
Spearman(U_2,E_2)
\]

\[
\rho_3
=
Spearman(U_3,E_3)
\]

越大越好。

不能只把所有 horizon 混起来计算一个 Spearman。

否则可能出现：

\[
horizon\uparrow
\Rightarrow
uncertainty\uparrow
\]

同时：

\[
horizon\uparrow
\Rightarrow
error\uparrow
\]

从而制造虚假的 uncertainty-error correlation。

### 15.2 Macro Spearman

定义：

\[
\rho_{\text{macro}}
=
\frac{
\rho_1+\rho_2+\rho_3
}{3}
\]

作为 Ensemble uncertainty 的主要总结指标。

### 15.3 High vs Low Uncertainty Error Ratio

由于数据量有限，不再使用 quintile。

改用：

**Top / Bottom Tertile**

即：

\[
R_k
=
\frac{
Error(\text{highest 33\% uncertainty})
}{
Error(\text{lowest 33\% uncertainty})
}
\]

希望：

\[
R_k>1
\]

但该指标只作为直观辅助指标，不作为唯一统计判据。

### 15.4 Uncertainty vs Horizon

额外报告：

\[
Spearman(U,k)
\]

它只作为 diagnostic metric。

用于判断 uncertainty 是否主要反映：

> rollout 越长，因此 uncertainty 越高

而不是：

> 当前 trajectory 本身更加难预测。

---

## 16. Statistical Testing

所有正式 comparison 使用：

**patient-level paired bootstrap**

设置：

```text
bootstrap_samples = 2000
```

对于每个患者先聚合该患者对应指标，再以患者为单位 bootstrap。

禁止把同一患者多个 window 当作完全独立样本。

### 16.1 RHRT Bootstrap

重点计算：

\[
\Delta MSE@2
=
MSE@2_{\text{baseline}}
-
MSE@2_{\text{RHRT}}
\]

\[
\Delta MSE@3
=
MSE@3_{\text{baseline}}
-
MSE@3_{\text{RHRT}}
\]

以及：

\[
\Delta MSE_{\text{long}}
\]

和：

\[
\Delta slope
=
slope_{\text{baseline}}
-
slope_{\text{RHRT}}
\]

均报告：

- mean improvement
- relative improvement
- 95% bootstrap CI
- probability of improvement

---

## 17. RQ1 通过标准

Random-Horizon Recursive Training 视为获得有效证据，需要：

### Criterion A

\[
MSE_{\text{long}}^{RHRT}
<
MSE_{\text{long}}^{baseline}
\]

且 patient-level bootstrap：

\[
95\%CI(\Delta MSE_{\text{long}})
\]

完全大于 0。

### Criterion B

matched H3 cohort 上：

\[
slope_{RHRT}<slope_{baseline}
\]

且 bootstrap improvement 为正。

同时报告：

\[
MSE@2
\]

和：

\[
MSE@3
\]

方便观察 improvement 来自哪个 horizon。

---

## 18. RQ2 通过标准

Ensemble reliability 视为获得有效证据，需要：

\[
\rho_{\text{macro}}>0
\]

同时 patient-level bootstrap 95% CI 下界大于 0。

辅助要求：

\[
R_{\text{macro}}>1
\]

其中：

\[
R_{\text{macro}}
=
\frac{R_1+R_2+R_3}{3}
\]

这里验证的是：

**uncertainty ranking ability**

而不是：

**probabilistic calibration**

因此正式论文中不要写：

> uncertainty is calibrated

应写：

> ensemble disagreement provides a useful ranking signal for prediction risk.

---

## 19. Random Seeds

主实验运行：

```text
7
17
29
```

单模型：

```text
baseline
rhrt
```

分别运行 3 次。

Ensemble：

```text
ensemble
rhrt_ensemble
```

每个 experimental seed 下包含 5 个独立 member。

---

## 20. Checkpoint Selection

所有模型使用相同 validation criterion。

建议使用：

\[
ValRecursiveMSE_{2:3}
=
\frac{
ValMSE@2+ValMSE@3
}{2}
\]

选择 best checkpoint。

这样 baseline 和 RHRT 都根据同一个 downstream rollout objective 选择模型。

不得：

- baseline 用 one-step validation loss
- RHRT 用 long-horizon validation loss

否则 checkpoint selection 不公平。

---

## 21. 输出文件重新设计

目录建议：

```text
outputs/stage1/
│
├── brainiac/
│   │
│   ├── seed_7/
│   │   ├── baseline/
│   │   │   ├── training.json
│   │   │   └── recursive_metrics.json
│   │   │
│   │   ├── rhrt/
│   │   │   ├── training.json
│   │   │   └── recursive_metrics.json
│   │   │
│   │   ├── ensemble/
│   │   │   ├── training.json
│   │   │   ├── recursive_metrics.json
│   │   │   └── uncertainty_metrics.json
│   │   │
│   │   └── rhrt_ensemble/
│   │       ├── training.json
│   │       ├── recursive_metrics.json
│   │       └── uncertainty_metrics.json
│   │
│   ├── seed_17/
│   ├── seed_29/
│   │
│   └── reports/
│       ├── rhrt_summary.json
│       ├── rhrt_bootstrap.json
│       ├── ensemble_summary.json
│       ├── ensemble_bootstrap.json
│       └── stage1_summary.json
│
└── mri_core/
    └── ...
```

---

## 22. 各文件职责

### `training.json`

只保存训练相关信息：

```text
variant
seed
best_epoch
train_loss
validation_loss
num_training_windows
horizon_sampling_counts
checkpoint
```

其中必须记录：

```text
k=1 sample count
k=2 sample count
k=3 sample count
```

用于检查 Random-Horizon Sampling 是否严重失衡。

### `recursive_metrics.json`

只保存 prediction metrics：

```text
horizon
n_predictions
n_patients
mse
cosine_distance
```

例如：

```text
horizon = 1
horizon = 2
horizon = 3
```

以及：

```text
long_horizon_mse
matched_h3_slope
```

不保存 uncertainty 信息。

### `uncertainty_metrics.json`

只保存 Ensemble reliability metrics：

```text
horizon
mean_uncertainty
uncertainty_error_spearman
high_low_tertile_error_ratio
```

以及：

```text
macro_spearman
macro_high_low_ratio
uncertainty_horizon_spearman
```

不保存 prediction bootstrap。

### `rhrt_bootstrap.json`

只保存 RHRT statistical tests：

```text
MSE@2 improvement
MSE@3 improvement
long-horizon improvement
matched-H3 slope improvement
95% CI
probability of improvement
```

### `ensemble_bootstrap.json`

只保存 uncertainty statistical tests：

```text
rho@1
rho@2
rho@3
macro rho
95% CI
```

### `stage1_summary.json`

只保留最终最关键结果：

```text
rhrt_pass
ensemble_pass
stage1_pass
```

以及少量核心数字：

```text
long_horizon_relative_improvement
recursive_slope_relative_improvement
macro_uncertainty_spearman
```

禁止再次复制所有 per-horizon metrics。

---

## 23. 终端指标打印

运行结束时，不直接 dump JSON。

使用表格打印。

### RHRT

```text
Random-Horizon Recursive Training

| Variant  | MSE@1 | MSE@2 | MSE@3 | Long MSE | H3 Slope |
|----------|------:|------:|------:|---------:|---------:|
| Baseline |       |       |       |          |          |
| RHRT     |       |       |       |          |          |
```

### RHRT Statistical Test

```text
| Metric        | Improvement | Relative | 95% CI | P(Improve) |
|---------------|------------:|---------:|--------|-----------:|
| MSE@2         |             |          |        |            |
| MSE@3         |             |          |        |            |
| Long MSE      |             |          |        |            |
| H3 Slope      |             |          |        |            |
```

### Ensemble Reliability

```text
| Horizon | Spearman ρ | High/Low Tertile Ratio |
|--------:|-----------:|------------------------:|
| 1       |            |                         |
| 2       |            |                         |
| 3       |            |                         |
| Macro   |            |                         |
```

这些表格同时写入 `.log` 即可。

不生成 CSV。

---

## 24. 推荐配置

```json
{
  "max_horizon": 3,
  "recursive_training": true,
  "terminal_loss_only": true,
  "teacher_forcing": false,
  "ensemble_size": 5,
  "split_ratio": [0.70, 0.15, 0.15],
  "split_seed": 17,
  "experiment_seeds": [7, 17, 29],
  "bootstrap_samples": 2000,
  "primary_metric": "normalized_latent_mse",
  "checkpoint_metric": "val_recursive_mse_k2_k3"
}
```

其余：

- optimizer
- learning rate
- batch size
- hidden dimension
- epoch number
- scheduler

保持现有 Stage 1 配置不变。

避免同时修改 dynamics architecture 和 training strategy。

---

## 25. 建议 CLI

修改后的接口建议拆成：

```bash
clarity-hauwm train-stage1 \
  --data data/trajectories/brainiac \
  --config configs/stage1_recursive.json \
  --variants baseline rhrt ensemble rhrt_ensemble \
  --seeds 7 17 29 \
  --output outputs/stage1/brainiac
```

训练完成后：

```bash
clarity-hauwm evaluate-recursive \
  --input outputs/stage1/brainiac \
  --max-horizon 3
```

然后：

```bash
clarity-hauwm evaluate-uncertainty \
  --input outputs/stage1/brainiac \
  --max-horizon 3
```

最后：

```bash
clarity-hauwm summarize-stage1 \
  --input outputs/stage1/brainiac \
  --bootstrap-samples 2000
```

这样：

训练、prediction evaluation、uncertainty evaluation 和 statistical testing 相互独立。

---

## 26. 最终实验逻辑

整个 Stage 1 可以压缩为：

```text
真实 MRI
   ↓
Frozen Encoder
   ↓
真实 z_t
   │
   │真实 treatment A_t
   ▼
Dynamics
   ↓
ẑ_t+1
   │
   │真实 treatment A_t+1
   ▼
Dynamics
   ↓
ẑ_t+2
   │
   │真实 treatment A_t+2
   ▼
Dynamics
   ↓
ẑ_t+3
```

训练时：

```text
随机选择 k = 1 / 2 / 3
              ↓
recursive rollout k steps
              ↓
仅在 z_t+k 计算监督
```

实验 1：

```text
Baseline
      vs
Random-Horizon Recursive Training
              ↓
Recursive MSE@1/2/3
Long-Horizon MSE
Matched-H3 Error Growth
```

实验 2：

```text
Independent Dynamics Ensemble
              ↓
Member-specific Recursive Rollout
              ↓
Ensemble Disagreement
              ↓
Prediction Error
              ↓
Within-Horizon Spearman
High/Low Uncertainty Error Ratio
```

---

## 27. Stage 1 最终需要回答的问题

如果实验成功，应能够分别支持两个结论。

### Random-Horizon Recursive Training

> Random-horizon recursive training improves multi-step patient-state prediction and reduces error accumulation compared with conventional one-step training.

### Ensemble

> Ensemble disagreement provides a useful signal for identifying unreliable latent-state predictions during recursive rollout.

这两个结论成立以后，下一阶段才有充分理由加入：

\[
Policy
\rightarrow
Candidate\ Treatments
\rightarrow
World\ Model
\rightarrow
Reliability
\rightarrow
Outcome
\rightarrow
Planning
\]

Stage 1 本身仍然不做治疗推荐。
