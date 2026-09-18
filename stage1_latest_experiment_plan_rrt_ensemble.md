# Stage 1 最新实验修改方案：Recursive Rollout Training + Ensemble Dynamics

## 0. 本次修改点

本次修改基于当前 Stage 1 已有实验结果，删除 Random-Horizon Sampling，将方法重新定义为 **Recursive Rollout Training (RRT) + Ensemble Dynamics**。

主要修改如下：

1. **删除 Random-Horizon Sampling**
   - 不再使用 `random_available`。
   - 不再使用 RHRT / Random-Horizon Recursive Training。
   - 不再研究 Random Horizon 是否优于 maximum horizon。

2. **将原 `recursive_max` 升格为正式方法 RRT**
   - 新方法名称：**Recursive Rollout Training (RRT)**。
   - 对每个训练起点使用当前样本可支持的最大 rollout horizon。
   - 主实验仍限制：
     \[
     K_{\max}=3.
     \]
   - 训练阶段递归回灌模型自己的 predicted latent state。

3. **Stage 1 最终保留两个核心模块**
   - **RRT**：提高 multi-step rollout accuracy。
   - **Ensemble Dynamics**：提供 trajectory-level reliability / prediction risk signal。

4. **主实验继续保持 \(K_{\max}=3\)**
   - 不因为删除 Random Horizon 而直接恢复到 5。
   - 主实验仍只评价 H1、H2、H3。
   - H4/H5 样本更少，更容易导致 prediction 和 reliability 指标高方差。

5. **新增 Training Horizon Ablation**
   - 正式研究：
     \[
     K_{\max}\in\{1,2,3\}.
     \]
   - 如果 H4/H5 样本量足够，再增加：
     \[
     K_{\max}=5
     \]
     作为 long-horizon stress test。
   - \(K_{\max}=5\) 不作为新的默认主实验设置。

6. **删除原 `recursive_max` ablation**
   - 因为 `recursive_max` 已成为正式 RRT。
   - 原来的 Baseline → Recursive-Max 直接改为 Baseline → RRT。

7. **实验 variants 从 5 个简化为 4 个**
   - `baseline`
   - `rrt`
   - `ensemble`
   - `rrt_ensemble`

8. **RQ1 改为**
   > Does Recursive Rollout Training improve multi-step recursive patient dynamics prediction?

9. **RQ2 保持**
   > Does ensemble disagreement provide a useful signal for identifying unreliable imagined trajectories?

10. **预测指标保持不变**
    - MSE@1
    - MSE@2
    - MSE@3
    - Long MSE
    - Relative Improvement
    - multi-seed mean ± std
    - repeated patient split robustness

11. **Ensemble reliability 指标保持不变**
    - within-horizon Spearman correlation
    - High vs Low disagreement error ratio
    - Selective Risk
    - 不称为 calibrated uncertainty。

12. **移除 horizon conditioning**
    - dynamics transition 统一学习：
      \[
      (z_t,A_t,\Delta t_t)\rightarrow z_{t+1}.
      \]
    - 如果当前代码中的 `horizon_embed_dim` 仅用于 target-horizon conditioning，则删除对应输入和 embedding。
    - Baseline 与 RRT 使用同一个 transition model architecture，核心差异仅为训练时是否 recursive rollout。

13. **保留 terminal-only supervision**
    - 当前阶段不同时修改 supervision strategy。
    - 每个 rollout 只对最终状态计算 loss。

14. **BrainIAC 保持主 Encoder**
    - BrainIAC：main experiment。
    - MRI-CORE：representation robustness。
    - 不直接比较两个 latent space 的绝对 MSE。

15. **继续不使用 95% CI、H3 slope 和自动 Pass / Fail**
    - 程序只输出实验事实。

---

# 1. Stage 1 目标

## RQ1

**Does Recursive Rollout Training improve multi-step recursive patient dynamics prediction?**

目标：

\[
\boxed{
\text{Conventional Transition Training}
\rightarrow
\text{RRT}
\rightarrow
\text{Lower Multi-Step Rollout Error}
}
\]

## RQ2

**Does ensemble disagreement provide a useful signal for identifying unreliable imagined trajectories?**

目标：

\[
\boxed{
\text{Ensemble Disagreement}
\rightarrow
\text{Trajectory Prediction Risk Signal}
}
\]

Stage 1 不研究：

- Policy model
- Treatment recommendation
- Treatment ranking
- Survival model
- Outcome optimization
- Counterfactual treatment effect
- Closed-loop planning

这些模块留到后续 Reliability-Aware Planning 阶段。

---

# 2. Stage 1 整体逻辑

\[
\boxed{
\text{Recursive Rollout Training}
\rightarrow
\text{More Accurate Multi-Step Dynamics}
}
\]

以及：

\[
\boxed{
\text{Ensemble Dynamics}
\rightarrow
\text{Trajectory Reliability Signal}
}
\]

最终形成：

\[
\boxed{
\text{Accurate Dynamics}
+
\text{Reliable Rollout Estimation}
}
\]

并为：

\[
\boxed{
\text{Reliability-Aware Planning}
}
\]

提供基础。

---

# 3. 数据与 Patient State

数据集：

`MU-Glioma-Post`

患者 trajectory：

\[
z_1,z_2,\ldots,z_T
\]

对应真实 treatment：

\[
A_1,A_2,\ldots,A_{T-1}
\]

时间间隔：

\[
\Delta t_1,\Delta t_2,\ldots,\Delta t_{T-1}.
\]

其中：

- \(z_t\)：MRI latent patient state
- \(A_t\)：\(t\rightarrow t+1\) 期间真实 treatment
- \(\Delta t_t\)：相邻 MRI timepoints 时间差

保持：

```text
action_anchor = source
```

避免 future treatment leakage。

---

# 4. Encoder 设置

## 4.1 BrainIAC

BrainIAC 作为主实验 Encoder。

```text
freeze_encoder = true
```

不使用：

- LoRA
- Encoder fine-tuning
- joint encoder-dynamics training

Dynamics model 学习：

\[
(z_t,A_t,\Delta t_t)\rightarrow z_{t+1}.
\]

## 4.2 MRI-CORE

MRI-CORE 作为 Representation Robustness Experiment。

BrainIAC 与 MRI-CORE：

- 分别训练 dynamics model
- 不共享 dynamics parameters
- 不直接比较绝对 MSE

重点比较每个 latent space 内：

\[
Baseline\rightarrow RRT
\]

的 relative improvement。

---

# 5. Patient-Level Data Split

| Split | Ratio |
|---|---:|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

主实验：

```text
split_seed = 17
```

同一患者所有 timepoints 必须位于同一个 split。

禁止：

- transition-level random split
- timepoint-level random split

所有 variants 共享：

- train patients
- validation patients
- test patients
- evaluation windows
- treatment sequence
- latent normalization

Latent normalization 只能由 training patients 计算。

---

# 6. Horizon 设计

## 6.1 主实验 Horizon

主实验保持：

\[
K_{\max}=3.
\]

对于训练起点 \(t\)：

\[
K_t=\min(K_{\max},T-t)=\min(3,T-t).
\]

正式主实验只评价：

\[
k=1,2,3.
\]

## 6.2 为什么不直接恢复 \(K_{\max}=5\)

增加 horizon 会同时带来：

1. 更强的 recursive error accumulation；
2. 更少的可用 training windows；
3. 更少的 test windows；
4. 更困难的 terminal-only optimization；
5. reliability metrics 更高的方差；
6. 更大的累计真实时间跨度差异。

因此主实验继续使用 H1-H3，以保证样本规模和 evaluation stability。

## 6.3 Horizon 不等于固定临床时间

因为模型显式输入：

\[
\Delta t_t,
\]

所以 H3 代表三个 observation transitions，并不代表固定三个月。

不同患者的：

\[
\sum_{j=0}^{k-1}\Delta t_{t+j}
\]

可能不同。

---

# 7. Recursive Rollout Training

对于训练起点 \(t\)：

\[
K_t=\min(3,T-t),
\]

始终使用：

\[
k=K_t.
\]

初始化：

\[
\hat z_t=z_t.
\]

递归：

\[
\hat z_{t+1}=f_\theta(z_t,A_t,\Delta t_t),
\]

\[
\hat z_{t+2}=f_\theta(\hat z_{t+1},A_{t+1},\Delta t_{t+1}),
\]

直到：

\[
\hat z_{t+k}.
\]

即：

\[
z_t
\rightarrow
\hat z_{t+1}
\rightarrow
\hat z_{t+2}
\rightarrow
\cdots
\rightarrow
\hat z_{t+k}.
\]

关键约束：

- 第一步输入真实 \(z_t\)
- 后续全部输入模型自己的 predicted latent
- treatment sequence 使用真实历史 treatment
- 使用真实 \(\Delta t\)
- 不使用 teacher forcing

```text
teacher_forcing = false
```

---

# 8. RRT Loss

采用 terminal-only supervision：

\[
\mathcal L_{\mathrm{RRT}}
=
\frac{1}{d}
\left\|
\hat z_{t+k}-z_{t+k}
\right\|_2^2.
\]

不对中间状态单独增加 loss。

```text
terminal_loss_only = true
max_horizon = 3
```

这样当前实验只改变 rollout training strategy，不同时引入新的 supervision design。

---

# 9. RRT Contribution 定位

推荐表述：

> **Recursive Rollout Training reduces the training–rollout mismatch in latent patient dynamics by recursively feeding model-predicted patient states back into the dynamics model during training.**

核心问题：

\[
\boxed{
\text{Training State Distribution}
\neq
\text{Inference Rollout State Distribution}
}
\]

RRT 的目的，是让模型在训练阶段显式接触由自身 prediction error 产生的 rollout states。

---

# 10. 实验 Variants

| Variant | Training Horizon | Recursive Feedback | Ensemble |
|---|---|---|---|
| Baseline | \(k=1\) | No | No |
| RRT | max available, capped at 3 | Yes | No |
| Ensemble | \(k=1\) | No | Yes |
| RRT + Ensemble | max available, capped at 3 | Yes | Yes |

删除：

```text
recursive_max
rhrt
rhrt_ensemble
random_available
```

改为：

```text
baseline
rrt
ensemble
rrt_ensemble
```

---

# 11. Baseline

\[
\hat z_{t+1}=f_\theta(z_t,A_t,\Delta t_t)
\]

Loss：

\[
\mathcal L_{\mathrm{base}}
=
\frac1d
\left\|
\hat z_{t+1}-z_{t+1}
\right\|_2^2.
\]

训练阶段：

\[
k=1.
\]

测试阶段仍统一执行 recursive rollout。

Baseline 与 RRT 使用完全相同的：

- architecture
- inputs
- optimizer
- training budget
- checkpoint selection
- evaluation windows

主要差异为训练时是否 recursive rollout。

---

# 12. Training Seeds

```text
training_seeds = [7, 17, 29]
```

每个 variant 运行三个 seeds，并报告：

\[
mean\pm std.
\]

---

# 13. RQ1 Primary Metrics

\[
MSE@1,\quad MSE@2,\quad MSE@3
\]

以及：

\[
MSE_{\mathrm{long}}
=
\frac{MSE@2+MSE@3}{2}.
\]

Long MSE 作为主要 summary metric。

---

# 14. Relative Improvement

\[
RI_k
=
\frac{
MSE_k^{baseline}-MSE_k^{method}
}{
MSE_k^{baseline}
}
\times100\%.
\]

报告：

\[
RI@1,\quad RI@2,\quad RI@3,\quad RI_{\mathrm{long}}.
\]

主要比较：

\[
\boxed{Baseline\rightarrow RRT}
\]

以及：

\[
\boxed{Ensemble\rightarrow RRT+Ensemble}.
\]

---

# 15. RRT Main Result Table

| Method | MSE@1 ↓ | MSE@2 ↓ | MSE@3 ↓ | Long MSE ↓ |
|---|---:|---:|---:|---:|
| Baseline | mean ± std | mean ± std | mean ± std | mean ± std |
| RRT | | | | |
| Ensemble | | | | |
| RRT + Ensemble | | | | |

---

# 16. Relative Improvement Table

| Comparison | RI@1 | RI@2 | RI@3 | RI Long |
|---|---:|---:|---:|---:|
| Baseline → RRT | | | | |
| Ensemble → RRT + Ensemble | | | | |

---

# 17. RRT 结果分析原则

不设置硬阈值。

重点观察：

### Overall Improvement

\[
MSE_{\mathrm{long}}^{RRT}
<
MSE_{\mathrm{long}}^{Baseline}.
\]

### Multi-Step Improvement

\[
MSE@2_{RRT}<MSE@2_{Baseline},
\]

\[
MSE@3_{RRT}<MSE@3_{Baseline}.
\]

### Seed Consistency

| Seed | Baseline Long | RRT Long | Relative Improvement |
|---:|---:|---:|---:|
| 7 | | | |
| 17 | | | |
| 29 | | | |

重点观察 improvement direction。

---

# 18. Patient Split Robustness

主 split：

```text
split_seed = 17
```

新增：

```text
robustness_split_seeds = [23, 41, 59]
```

固定：

```text
training_seed = 17
```

只运行：

```text
baseline
rrt
```

输出：

| Split Seed | Baseline Long MSE | RRT Long MSE | RRT Relative Improvement |
|---:|---:|---:|---:|
| 17 | | | |
| 23 | | | |
| 41 | | | |
| 59 | | | |

用于判断 RRT improvement 是否依赖某个特定 patient split。

---

# 19. Secondary Prediction Metric

\[
CosDist
=
1-
\frac{
\hat z^\top z
}{
\|\hat z\|_2\|z\|_2
}.
\]

报告：

\[
Cos@1,\quad Cos@2,\quad Cos@3.
\]

仅作为 secondary metric。

---

# 20. Ensemble Dynamics

Ensemble size：

\[
M=5.
\]

构建：

\[
f_{\theta_1},f_{\theta_2},\ldots,f_{\theta_5}.
\]

每个 member：

- 独立 initialization
- 独立 random seed
- 独立 mini-batch shuffle
- RRT setting 下独立 recursive rollout

每个 member 维护自己的：

\[
\hat z_{t+j}^{(m)}.
\]

禁止先计算 ensemble mean，再将 mean 回灌到下一步。

---

# 21. Ensemble Prediction

\[
\bar z_{t+k}
=
\frac1M
\sum_{m=1}^{M}
\hat z_{t+k}^{(m)}.
\]

Prediction error：

\[
E_{t+k}
=
\frac1d
\left\|
\bar z_{t+k}-z_{t+k}
\right\|_2^2.
\]

---

# 22. Ensemble Disagreement

\[
U_{t+k}
=
\frac1M
\sum_{m=1}^{M}
\frac1d
\left\|
\hat z_{t+k}^{(m)}
-
\bar z_{t+k}
\right\|_2^2.
\]

不称其为：

```text
calibrated uncertainty
```

统一使用：

- ensemble disagreement
- trajectory reliability signal
- prediction risk signal

---

# 23. RQ2

验证：

\[
U\uparrow
\Rightarrow
E\uparrow.
\]

---

# 24. Within-Horizon Spearman

分别计算：

\[
\rho_k=Spearman(U_k,E_k).
\]

得到：

\[
\rho_1,\quad\rho_2,\quad\rho_3.
\]

禁止混合 H1/H2/H3 后直接计算 correlation。

定义：

\[
\rho_{\mathrm{macro}}
=
\frac{\rho_1+\rho_2+\rho_3}{3}.
\]

---

# 25. High vs Low Disagreement Error

每个 horizon 内按 disagreement 分为：

- Lowest 33%
- Middle 33%
- Highest 33%

计算：

\[
R_k
=
\frac{Error_{High}}{Error_{Low}}.
\]

若：

\[
R_k>1,
\]

表示 high-disagreement trajectories 的 prediction error 更高。

---

# 26. Selective Risk

对每个 horizon：

1. 按 \(U\) 从高到低排序；
2. 计算全部 trajectories 的 \(Risk_{100}\)；
3. 删除 disagreement 最高的 20%；
4. 计算剩余 80% 的 \(Risk_{80}\)。

定义：

\[
RiskReduction@80
=
\frac{Risk_{100}-Risk_{80}}{Risk_{100}}
\times100\%.
\]

若：

\[
Risk_{80}<Risk_{100},
\]

说明 disagreement 可以帮助过滤高风险 imagined trajectories。

---

# 27. Ensemble Result Table

| Method | ρ@1 | ρ@2 | ρ@3 | Macro ρ | Macro High/Low Ratio |
|---|---:|---:|---:|---:|---:|
| Ensemble | | | | | |
| RRT + Ensemble | | | | | |

Selective Risk：

| Horizon | Risk@100% | Risk@80% | Risk Reduction |
|---:|---:|---:|---:|
| H1 | | | |
| H2 | | | |
| H3 | | | |

---

# 28. Training Horizon Ablation

删除 Random-Horizon ablation 后，新增：

\[
K_{\max}\in\{1,2,3\}.
\]

其中：

- \(K_{\max}=1\)：conventional one-step training
- \(K_{\max}=2\)：最多递归 2 步
- \(K_{\max}=3\)：正式 RRT setting

目的：

> How does the amount of recursive rollout exposure during training affect multi-step prediction?

统一使用相同 H1-H3 test windows：

| Training Max Horizon | MSE@1 | MSE@2 | MSE@3 | Long MSE |
|---|---:|---:|---:|---:|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

推荐先固定：

```text
split_seed = 17
training_seed = 17
```

作为低成本 ablation。

如果需要进一步 robustness，再使用：

```text
training_seeds = [7,17,29]
```

重复。

---

# 29. Optional H5 Stress Test

只有 Data Audit 显示 H4/H5 的 train/val/test windows 足够时，才增加：

\[
K_{\max}=5.
\]

定位：

```text
long-horizon stress test
```

而不是新的主实验 setting。

训练 horizon：

\[
K_{\text{train}}\in\{1,2,3,5\}.
\]

主比较仍统一报告：

\[
MSE@1,\quad MSE@2,\quad MSE@3.
\]

如果 H4/H5 test windows 足够，再描述性报告：

\[
MSE@4,\quad MSE@5.
\]

H4/H5 不并入主：

\[
MSE_{\mathrm{long}}
=
\frac{MSE@2+MSE@3}{2}.
\]

H4/H5 reliability 只有在每个 horizon 的样本量足够时才计算。

---

# 30. Horizon Ablation 解释原则

### Case A

\[
K=1>K=2>K=3
\]

其中 \(>\) 表示 MSE 更高。

说明：

> Longer recursive exposure progressively improves multi-step prediction.

### Case B

\[
K=2\approx K=3
\]

说明：

> Most of the benefit is obtained from short recursive rollout exposure.

### Case C

\[
K=3<K=5
\]

说明：

> Increasing the training horizon beyond three steps does not necessarily provide additional benefit.

不预设 \(K_{\max}=5\) 一定优于 \(K_{\max}=3\)。

---

# 31. Checkpoint Selection

统一使用：

\[
ValScore
=
\frac{MSE@2_{val}+MSE@3_{val}}{2}.
\]

选择：

\[
\theta^*=\arg\min_\theta ValScore.
\]

```text
checkpoint_metric = val_recursive_mse_k2_k3
```

所有 variants 使用完全相同的模型选择规则。

---

# 32. Training Hyperparameters

除 rollout strategy 外，各 variants 保持一致：

```text
batch_size = 32
epochs = 100
learning_rate = 1e-3
weight_decay = 1e-4

hidden_dim = 128
action_embed_dim = 32
time_embed_dim = 16

gradient_clip_norm = 1.0
early_stopping_patience = 15

max_horizon = 3
ensemble_size = 5
```

删除：

```text
horizon_embed_dim
```

前提是当前该参数仅用于 target-horizon conditioning。

Transition input 统一为：

```text
latent_state
treatment
delta_time
```

---

# 33. Config 修改

删除：

```text
random_available
rhrt
rhrt_ensemble
recursive_max
horizon_embed_dim
```

新增或保留：

```text
main_split_seed = 17
training_seeds = [7,17,29]

robustness_split_seeds = [23,41,59]
robustness_training_seed = 17

main_max_horizon = 3
horizon_ablation = [1,2,3]
optional_stress_horizon = 5

ensemble_size = 5
```

---

# 34. Horizon Strategy 重构

只保留：

```text
one_step
max_available
```

对应：

```text
Baseline -> one_step
RRT      -> max_available
```

### one_step

\[
k=1.
\]

### max_available

\[
k=\min(K_{\max},T-t).
\]

---

# 35. `data.py` 修改

支持：

```text
horizon_strategy
max_horizon
```

### one_step

```text
k = 1
```

### max_available

```text
K_t = min(max_horizon, T - t)
k = K_t
```

删除：

```text
random_available
epoch-based horizon resampling
horizon_sampling_counts
```

改为记录：

```text
available_horizon_counts
training_horizon_counts
```

用于确认不同 \(K_{\max}\) 下实际 H1/H2/H3 训练样本数量。

---

# 36. `training.py` 修改

Variants：

```text
baseline
rrt
ensemble
rrt_ensemble
```

对应：

```text
baseline:
    horizon_strategy = one_step
    ensemble = false

rrt:
    horizon_strategy = max_available
    ensemble = false

ensemble:
    horizon_strategy = one_step
    ensemble = true

rrt_ensemble:
    horizon_strategy = max_available
    ensemble = true
```

RRT：

```text
teacher_forcing = false
terminal_loss_only = true
max_horizon = 3
```

Horizon ablation 通过覆盖：

```text
max_horizon
```

完成，不新增 method variant。

---

# 37. `model.py` 修改

Dynamics model 输入统一：

\[
(z_t,A_t,\Delta t_t).
\]

如果当前 forward 包含：

```text
horizon
horizon_embedding
```

则删除。

目标：

\[
f_\theta:(z_t,A_t,\Delta t_t)\rightarrow\hat z_{t+1}.
\]

RRT 的 multi-step ability 来自将：

\[
\hat z_{t+j}
\]

递归输入同一个 \(f_\theta\)，而不是 horizon conditioning。

---

# 38. `reporting.py` 修改

Prediction：

```text
MSE@1
MSE@2
MSE@3
Long MSE
mean
std
Relative Improvement
```

Secondary：

```text
Cos@1
Cos@2
Cos@3
```

Reliability：

```text
rho@1
rho@2
rho@3
macro_rho

high_low_ratio@1
high_low_ratio@2
high_low_ratio@3
macro_high_low_ratio

selective_risk
```

新增：

```text
horizon_ablation
```

删除：

```text
Random Horizon statistics
RHRT comparison
Recursive-Max comparison
horizon sampling distribution
```

---

# 39. 输出目录

```text
outputs/
└── stage1/
    └── <encoder>/
        ├── seed_7/
        ├── seed_17/
        ├── seed_29/
        ├── horizon_ablation/
        │   ├── k1/
        │   ├── k2/
        │   ├── k3/
        │   └── k5/        # optional
        └── reports/
            ├── dataset_stats.json
            ├── training_summary.json
            ├── prediction_metrics.json
            ├── prediction_comparison.json
            ├── uncertainty_metrics.json
            ├── selective_risk.json
            ├── split_robustness.json
            ├── horizon_ablation.json
            └── stage1_summary.json
```

不保存 CSV。

多指标结果在终端中用表格打印。

---

# 40. 各结果文件内容

## `dataset_stats.json`

```text
train_patient_count
val_patient_count
test_patient_count

H1_window_count
H2_window_count
H3_window_count

H4_window_count     # optional
H5_window_count     # optional

train_timepoint_count
val_timepoint_count
test_timepoint_count
```

## `training_summary.json`

```text
variant
training_seed
max_horizon

best_epoch
best_validation_score

H1_training_count
H2_training_count
H3_training_count

H4_training_count   # optional
H5_training_count   # optional
```

## `prediction_metrics.json`

四个 variants：

```text
baseline
rrt
ensemble
rrt_ensemble
```

保存：

```text
MSE@1
MSE@2
MSE@3
Long MSE

mean
std

Cos@1
Cos@2
Cos@3
```

## `prediction_comparison.json`

```text
baseline_vs_rrt
ensemble_vs_rrt_ensemble
```

每个 comparison：

```text
RI@1
RI@2
RI@3
RI_long
```

## `uncertainty_metrics.json`

保存 Ensemble 与 RRT+Ensemble：

```text
rho@1
rho@2
rho@3
macro_rho

high_low_ratio@1
high_low_ratio@2
high_low_ratio@3
macro_high_low_ratio
```

## `selective_risk.json`

按 method 和 horizon：

```text
risk_100
risk_80
risk_reduction
```

## `split_robustness.json`

```text
split_seed
baseline_long_mse
rrt_long_mse
rrt_relative_improvement
```

## `horizon_ablation.json`

```text
K1:
    MSE@1
    MSE@2
    MSE@3
    Long MSE

K2:
    MSE@1
    MSE@2
    MSE@3
    Long MSE

K3:
    MSE@1
    MSE@2
    MSE@3
    Long MSE

K5:                    # optional
    MSE@1
    MSE@2
    MSE@3
    MSE@4
    MSE@5
```

## `stage1_summary.json`

```text
main_encoder
main_split_seed
main_max_horizon

baseline_long_mse
rrt_long_mse
rrt_relative_improvement

ensemble_long_mse
rrt_ensemble_long_mse
rrt_ensemble_relative_improvement

macro_uncertainty_spearman
macro_high_low_ratio
risk_reduction_80
```

不保存自动 Pass / Fail。

---

# 41. 正式实验执行顺序

## Phase A：Data Audit

生成：

```text
dataset_stats.json
```

确认：

- Patient split
- H1/H2/H3 sample count
- H4/H5 sample count，仅用于决定是否运行 H5 stress test
- Encoder alignment
- Treatment alignment
- Delta time
- latent normalization source

## Phase B：BrainIAC Main Experiment

固定：

```text
split_seed = 17
max_horizon = 3
```

运行：

```text
baseline × seeds 7,17,29
rrt × seeds 7,17,29
ensemble × seeds 7,17,29
rrt_ensemble × seeds 7,17,29
```

## Phase C：BrainIAC Split Robustness

使用：

```text
split_seeds = [17,23,41,59]
training_seed = 17
max_horizon = 3
```

只运行：

```text
baseline
rrt
```

## Phase D：Training Horizon Ablation

固定：

```text
encoder = BrainIAC
split_seed = 17
training_seed = 17
```

运行：

```text
Kmax = 1
Kmax = 2
Kmax = 3
```

统一评价：

```text
MSE@1
MSE@2
MSE@3
Long MSE
```

## Phase E：Optional H5 Stress Test

先检查 H4/H5 window 数量。

只有样本数量足够时运行：

```text
Kmax = 5
```

定位：

```text
secondary long-horizon stress test
```

不改变主实验：

```text
Kmax = 3
```

## Phase F：MRI-CORE Representation Robustness

首先运行：

```text
baseline
rrt
rrt_ensemble
```

固定：

```text
Kmax = 3
```

重点比较：

\[
Baseline\rightarrow RRT
\]

的 within-representation relative improvement。

同时观察 RRT+Ensemble：

- Spearman
- High/Low Ratio
- Selective Risk

如需完整 reliability control，再补：

```text
ensemble
```

---

# 42. 最终实验问题

1. Conventional transition training 在 recursive rollout 下表现如何？
2. RRT 是否改善 H2/H3 multi-step patient state prediction？
3. RRT improvement 是否对 training initialization 稳定？
4. RRT improvement 是否依赖特定 patient split？
5. 不同 \(K_{\max}\) 的 recursive rollout exposure 对 multi-step prediction 有什么影响？
6. \(K_{\max}=3\) 是否已经获得主要收益？
7. 如果数据支持，\(K_{\max}=5\) 是否提供额外 long-horizon benefit？
8. RRT 是否能在不同 frozen MRI representations 上表现出类似趋势？
9. Ensemble disagreement 是否能够识别 prediction error 较大的 trajectory？
10. 根据 disagreement 过滤高风险 trajectories 后，剩余 rollout risk 是否下降？

---

# 43. 推荐 Contribution 表述

## Contribution 1：Recursive Rollout Training

> We introduce Recursive Rollout Training (RRT) for latent patient dynamics modeling. RRT recursively feeds model-predicted patient states back into the dynamics model during training, reducing the mismatch between transition training and multi-step inference rollouts.

重点：

\[
\boxed{\text{Training–Rollout Mismatch}}
\]

而不是声称 recursive learning 本身从未出现过。

## Contribution 2：Trajectory Reliability with Ensemble Dynamics

> We further construct ensemble patient dynamics and use trajectory-level model disagreement as a prediction risk signal, enabling unreliable imagined patient trajectories to be identified before downstream planning.

重点：

\[
\boxed{\text{Trajectory-Level Reliability}}
\]

而不是声称 ensemble 本身是全新的。

---

# 44. 最终 Stage 1 方法逻辑

删除：

\[
\text{Random Horizon}
+
\text{Recursive Training}.
\]

最终使用：

\[
\boxed{
\text{Recursive Rollout Training}
}
\]

核心：

\[
z_t
\rightarrow
\hat z_{t+1}
\rightarrow
\hat z_{t+2}
\rightarrow
\cdots
\]

以及：

\[
\boxed{
\text{Ensemble Dynamics}
\rightarrow
\text{Trajectory Reliability Estimation}
}
\]

最终形成：

\[
\boxed{
\text{RRT}
\rightarrow
\text{Accurate Multi-Step Patient Dynamics}
}
\]

和：

\[
\boxed{
\text{Ensemble Dynamics}
\rightarrow
\text{Reliable Trajectory Identification}
}
\]

进一步支持：

\[
\boxed{
\text{Accurate + Reliable Patient World Model}
\rightarrow
\text{Reliability-Aware Planning}
}
\]

---

# 45. 当前推荐的最小正式实验集合

## Main

```text
BrainIAC
split_seed = 17
training_seeds = [7,17,29]
Kmax = 3

baseline
rrt
ensemble
rrt_ensemble
```

## Split Robustness

```text
split_seeds = [17,23,41,59]
training_seed = 17

baseline
rrt
```

## Horizon Ablation

```text
Kmax = [1,2,3]
split_seed = 17
training_seed = 17
```

## Representation Robustness

```text
MRI-CORE
Kmax = 3

baseline
rrt
rrt_ensemble
```

## Optional

```text
Kmax = 5
```

仅作为 long-horizon stress test。

最终形成：

\[
\boxed{
\text{RRT Accuracy Evidence}
+
\text{Ensemble Reliability Evidence}
}
\]

而无需继续保留 Random-Horizon Sampling。
