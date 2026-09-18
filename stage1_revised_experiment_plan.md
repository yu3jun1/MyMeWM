# Stage 1 实验修改方案：Random-Horizon Recursive Training 与 Ensemble Dynamics

## 0. 本次修改点

本次修改基于当前 Stage 1 实验结果，重点简化统计判定逻辑，并将 Random-Horizon Sampling 与 Recursive Training 统一定义为一个完整方法：**Random-Horizon Recursive Training, RHRT**。

主要修改如下：

1. **将 Horizon Sampling 与 Recursive Training 合并为一个整体创新点 RHRT**
   - 不再将 Random Horizon 和 Recursive Training 分别作为两个 contribution。
   - RHRT 定义为：随机采样 rollout horizon，并在训练时递归回灌模型预测的 latent state。

2. **新增 `recursive_max` 作为 RHRT 的 ablation**
   - 始终使用当前样本可用的最大 horizon。
   - 用于分析 mixed random horizons 是否优于固定最长 horizon。
   - `recursive_max` 不作为独立 contribution。

3. **删除 95% Confidence Interval**
   - 不再使用 bootstrap 95% CI。
   - 不再通过 CI 是否跨 0 判断 Stage 1 是否成功。
   - 改为报告 MSE、relative improvement、multi-seed mean ± std 和 repeated patient split robustness。

4. **删除 H3 slope**
   - 不再使用 matched-H3 slope 衡量 error accumulation。
   - 不再将 slope improvement 作为 RHRT 是否有效的判断标准。
   - 直接比较 MSE@1、MSE@2、MSE@3 和 Long MSE。

5. **删除自动 Pass / Fail 判定**
   - 删除 `rhrt_pass`、`ensemble_pass` 和 `stage1_pass`。
   - 程序只输出实验结果，不自动给出 true / false 判断。

6. **重新定义 Stage 1 主指标**
   - MSE@1
   - MSE@2
   - MSE@3
   - Long MSE

\[
MSE_{\mathrm{long}}=\frac{MSE@2+MSE@3}{2}
\]

7. **增加 Relative Improvement**
   - 重点报告 Baseline → RHRT、Recursive-Max → RHRT、Ensemble → RHRT + Ensemble。

8. **使用 multi-seed mean ± std 替代复杂统计显著性检验**
   - 主实验训练 seeds：7、17、29。
   - 每个 seed 单独评估，再报告 mean ± std。

9. **新增 patient split robustness**
   - 主 split 保持 `split_seed = 17`。
   - 新增 `split_seed = 23, 41, 59`。
   - robustness experiment 固定 `training_seed = 17`。

10. **保留 BrainIAC 作为主 Encoder**
    - BrainIAC 为 Stage 1 main experiment。
    - MRI-CORE 为 representation robustness experiment。
    - 不直接比较两个 latent space 的绝对 MSE。

11. **Ensemble reliability 指标简化**
    - 保留 within-horizon Spearman correlation。
    - 保留 High vs Low uncertainty error ratio。
    - 新增 Selective Risk。
    - 删除 ensemble bootstrap CI。

12. **重新组织结果文件**
    - Prediction、uncertainty、split robustness、selective risk 分文件保存。
    - 不保存 CSV。
    - 多指标结果在终端中用表格打印。

---

# 1. Stage 1 目标

Stage 1 聚焦两个问题：

## RQ1

**Random-Horizon Recursive Training 是否能够改善 multi-step recursive patient dynamics prediction？**

## RQ2

**Ensemble disagreement 是否能够作为 imagined trajectory prediction risk 的有效信号？**

Stage 1 不研究：

- Policy model
- Treatment recommendation
- Treatment ranking
- Survival model
- Outcome optimization
- Counterfactual treatment effect
- Closed-loop planning

这些模块留到后续阶段。

---

# 2. Stage 1 整体逻辑

\[
\boxed{
\text{RHRT}
\rightarrow
\text{More Accurate Multi-Step Rollout}
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

两者共同为后续 Reliability-Aware Planning 提供基础。

---

# 3. 数据与 Patient State

数据集：

`MU-Glioma-Post`

患者 trajectory：

\[
z_1,z_2,\ldots,z_T
\]

对应 treatment：

\[
A_1,A_2,\ldots,A_{T-1}
\]

以及时间间隔：

\[
\Delta t_1,\Delta t_2,\ldots,\Delta t_{T-1}
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

BrainIAC 作为 Stage 1 主实验 Encoder。

Encoder 完全冻结：

```text
freeze_encoder = true
```

不使用 LoRA、Encoder fine-tuning 或 joint encoder-dynamics training。

Dynamics model 学习：

\[
(z_t,A_t,\Delta t_t)\rightarrow z_{t+1}
\]

## 4.2 MRI-CORE

MRI-CORE 作为 Representation Robustness Experiment。

BrainIAC 与 MRI-CORE：

- 分别训练 dynamics model
- 不共享 dynamics parameters
- 不直接比较绝对 MSE

重点比较每个 latent space 内：

\[
Baseline\rightarrow RHRT
\]

的 relative improvement。

---

# 5. Patient-Level Data Split

使用：

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

所有 variant 必须共享：

- train patients
- validation patients
- test patients
- evaluation windows
- treatment sequence
- latent normalization

Latent normalization 只能由 training patients 计算。

---

# 6. Horizon 设置

最大 horizon：

\[
K_{\max}=3
\]

对于训练起点 \(t\)：

\[
K_t=\min(3,T-t)
\]

正式实验只研究：

\[
k=1,2,3
\]

Stage 1 不考虑 Horizon 4 或 Horizon 5。

---

# 7. Random-Horizon Recursive Training

RHRT 将 Random-Horizon Sampling 与 Recursive Prediction Feedback 统一为一个完整训练机制。

对于每个训练起点：

\[
K_t=\min(3,T-t)
\]

随机采样：

\[
k\sim Uniform(1,K_t)
\]

初始化：

\[
\hat z_t=z_t
\]

然后递归：

\[
\hat z_{t+1}=f_\theta(z_t,A_t,\Delta t_t)
\]

\[
\hat z_{t+2}=f_\theta(\hat z_{t+1},A_{t+1},\Delta t_{t+1})
\]

继续直到：

\[
\hat z_{t+k}
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
\hat z_{t+k}
\]

关键约束：

- 第一步输入真实 \(z_t\)
- 后续全部输入模型自己的预测 latent
- Treatment sequence 使用真实历史 treatment
- 不使用 teacher forcing

```text
teacher_forcing = false
```

---

# 8. RHRT Loss

采用 terminal-only supervision：

\[
\mathcal L_{\mathrm{RHRT}}
=
\frac{1}{d}
\left\|
\hat z_{t+k}-z_{t+k}
\right\|_2^2
\]

不对中间预测状态分别增加 loss。

保持：

```text
terminal_loss_only = true
max_horizon = 3
```

---

# 9. RHRT Contribution 定位

论文中不将 Random Horizon 和 Recursive Training 拆成两个创新点。

统一表述：

> **Random-Horizon Recursive Training (RHRT)** randomly samples rollout horizons during training and recursively feeds predicted latent states back into the dynamics model, exposing the model to its own accumulated rollout errors over trajectories of varying lengths.

核心比较为：

\[
\boxed{
\text{Conventional One-Step Training}
\quad vs\quad
\text{RHRT}
}
\]

---

# 10. 实验 Variants

| Variant | Training Horizon | Recursive Feedback | Ensemble |
|---|---|---|---|
| Baseline | \(k=1\) | No | No |
| Recursive-Max | 最大可用 horizon | Yes | No |
| RHRT | Random horizon | Yes | No |
| Ensemble | \(k=1\) | No | Yes |
| RHRT + Ensemble | Random horizon | Yes | Yes |

---

# 11. Baseline

Baseline 使用 conventional one-step training：

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
\right\|_2^2
\]

训练阶段始终：

\[
k=1
\]

测试阶段仍统一执行 recursive rollout。

---

# 12. Recursive-Max Ablation

新增：

```text
recursive_max
```

对于每个训练起点：

\[
k=K_t
\]

例如：

```text
剩余 1 步 -> k=1
剩余 2 步 -> k=2
剩余 >=3 步 -> k=3
```

同样执行 recursive feedback，并使用 terminal-only loss。

Recursive-Max 仅作为 **RHRT Ablation**，不作为独立 contribution。

主要比较：

\[
Recursive\text{-}Max
\quad vs\quad
RHRT
\]

用于分析 mixed random horizons 是否优于始终训练最大 horizon。

---

# 13. Training Seeds

主实验使用：

```text
training_seeds = [7, 17, 29]
```

每个 variant 都运行三个 seed。

每个 seed 单独计算预测指标，最终报告：

\[
mean\pm std
\]

---

# 14. 删除 95% CI

删除：

```text
bootstrap_samples
paired bootstrap CI
CI lower bound
CI upper bound
P(improvement)
```

不再通过：

\[
95\%CI>0
\]

判断 Stage 1 是否成功。

---

# 15. 删除 H3 Slope

删除：

```text
matched_h3_slope
slope_improvement
```

不再根据 H1、H2、H3 拟合 error accumulation slope。

直接比较各 horizon prediction error。

---

# 16. 删除 Pass / Fail

删除：

```text
rhrt_pass
ensemble_pass
stage1_pass
```

程序只输出实验事实，不自动输出 true / false。

---

# 17. RQ1 Primary Metrics

保留：

\[
MSE@1
\]

\[
MSE@2
\]

\[
MSE@3
\]

定义：

\[
MSE_{\mathrm{long}}
=
\frac{MSE@2+MSE@3}{2}
\]

作为主要 summary metric。

---

# 18. Relative Improvement

\[
RI_k
=
\frac{
MSE_k^{baseline}-MSE_k^{method}
}{
MSE_k^{baseline}
}
\times100\%
\]

分别计算：

\[
RI@1,\quad RI@2,\quad RI@3,\quad RI_{\mathrm{long}}
\]

---

# 19. RHRT Main Result Table

| Method | MSE@1 ↓ | MSE@2 ↓ | MSE@3 ↓ | Long MSE ↓ |
|---|---:|---:|---:|---:|
| Baseline | mean ± std | mean ± std | mean ± std | mean ± std |
| Recursive-Max | | | | |
| RHRT | | | | |
| Ensemble | | | | |
| RHRT + Ensemble | | | | |

---

# 20. Relative Improvement Table

| Comparison | RI@1 | RI@2 | RI@3 | RI Long |
|---|---:|---:|---:|---:|
| Baseline → Recursive-Max | | | | |
| Baseline → RHRT | | | | |
| Recursive-Max → RHRT | | | | |
| Ensemble → RHRT + Ensemble | | | | |

最重要的是：

\[
Baseline\rightarrow RHRT
\]

其他比较主要用于 ablation 和解释。

---

# 21. RHRT 结果判断

不设置硬阈值。

重点观察：

## 21.1 Overall Improvement

\[
MSE_{\mathrm{long}}^{RHRT}
<
MSE_{\mathrm{long}}^{Baseline}
\]

## 21.2 Multi-Step Improvement

\[
MSE@2_{RHRT}<MSE@2_{Baseline}
\]

以及：

\[
MSE@3_{RHRT}<MSE@3_{Baseline}
\]

## 21.3 Seed Consistency

| Seed | Baseline Long | RHRT Long | Relative Improvement |
|---:|---:|---:|---:|
| 7 | | | |
| 17 | | | |
| 29 | | | |

观察不同 training seeds 的 improvement direction。

---

# 22. Patient Split Robustness

主实验：

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
recursive_max
rhrt
```

输出：

| Split Seed | Baseline Long MSE | Recursive-Max | RHRT | RHRT RI |
|---:|---:|---:|---:|---:|
| 17 | | | | |
| 23 | | | | |
| 41 | | | | |
| 59 | | | | |

用于判断 RHRT improvement 是否依赖特定 patient split。

---

# 23. Secondary Prediction Metric

保留 cosine distance：

\[
CosDist
=
1-
\frac{
\hat z^\top z
}{
\|\hat z\|_2\|z\|_2
}
\]

计算：

\[
Cos@1,\quad Cos@2,\quad Cos@3
\]

仅作为 secondary metric。

---

# 24. Ensemble Dynamics

Ensemble size：

\[
M=5
\]

构建：

\[
f_{\theta_1},
f_{\theta_2},
\ldots,
f_{\theta_5}
\]

每个 member：

- 独立 initialization
- 独立 random seed
- 独立 mini-batch shuffle
- RHRT 下独立 horizon sampling

每个 member 维护自己的 recursive rollout。

禁止先计算 ensemble mean，再将 mean 输入下一步。

---

# 25. Ensemble Prediction

\[
\bar z_{t+k}
=
\frac1M
\sum_{m=1}^{M}
\hat z_{t+k}^{(m)}
\]

Prediction error：

\[
E_{t+k}
=
\frac1d
\left\|
\bar z_{t+k}
-
z_{t+k}
\right\|_2^2
\]

---

# 26. Ensemble Disagreement

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
\right\|_2^2
\]

Stage 1 不称其为 calibrated uncertainty，而使用：

- trajectory reliability signal
- prediction risk signal

---

# 27. RQ2

> Does ensemble disagreement provide a useful signal for identifying unreliable imagined trajectories?

验证目标：

\[
U\uparrow
\Rightarrow
E\uparrow
\]

---

# 28. Within-Horizon Spearman

每个 horizon 单独计算：

\[
\rho_k=Spearman(U_k,E_k)
\]

得到：

\[
\rho_1,\quad\rho_2,\quad\rho_3
\]

禁止混合 H1、H2、H3 后直接计算 correlation。

定义：

\[
\rho_{\mathrm{macro}}
=
\frac{
\rho_1+\rho_2+\rho_3
}{3}
\]

---

# 29. High vs Low Uncertainty Error

每个 horizon 内按照 uncertainty 划分：

- Lowest 33%
- Middle 33%
- Highest 33%

计算：

\[
R_k
=
\frac{
Error_{High}
}{
Error_{Low}
}
\]

如果：

\[
R_k>1
\]

表示 high-disagreement trajectories 的 prediction error 更高。

---

# 30. Selective Risk

新增 Selective Risk。

对于每个 horizon：

1. 按照 \(U\) 从高到低排序。
2. 计算全部 trajectories 的 \(Risk_{100}\)。
3. 删除 uncertainty 最高的 20%。
4. 计算剩余 80% 的 \(Risk_{80}\)。

定义：

\[
RiskReduction@80
=
\frac{
Risk_{100}-Risk_{80}
}{
Risk_{100}
}
\times100\%
\]

如果：

\[
Risk_{80}<Risk_{100}
\]

说明 ensemble disagreement 可以帮助过滤高风险 imagined trajectories。

---

# 31. Ensemble Result Table

| Method | ρ@1 | ρ@2 | ρ@3 | Macro ρ | High/Low Ratio |
|---|---:|---:|---:|---:|---:|
| Ensemble | | | | | |
| RHRT + Ensemble | | | | | |

Selective Risk：

| Horizon | Risk@100% | Risk@80% | Risk Reduction |
|---:|---:|---:|---:|
| H1 | | | |
| H2 | | | |
| H3 | | | |

---

# 32. Horizon Sampling Distribution

每个 RHRT run 记录：

| Horizon | Count | Percentage |
|---|---:|---:|
| H1 | | |
| H2 | | |
| H3 | | |

Stage 1 暂时不修改 sampling strategy。

如果后续发现 H3 占比过低，再单独研究 Balanced Random-Horizon Sampling。

---

# 33. Checkpoint Selection

所有方法统一使用：

\[
ValScore
=
\frac{
MSE@2_{val}+MSE@3_{val}
}{2}
\]

选择：

\[
\theta^*
=
\arg\min_\theta ValScore
\]

保持：

```text
checkpoint_metric = val_recursive_mse_k2_k3
```

Baseline、Recursive-Max、RHRT 和 Ensemble variants 使用完全相同的模型选择规则。

---

# 34. Training Hyperparameters

除 horizon strategy 外，各 variant 保持一致：

```text
batch_size = 32
epochs = 100
learning_rate = 1e-3
weight_decay = 1e-4

hidden_dim = 128
action_embed_dim = 32
time_embed_dim = 16
horizon_embed_dim = 16

gradient_clip_norm = 1.0
early_stopping_patience = 15

max_horizon = 3
ensemble_size = 5
```

---

# 35. Config 修改

删除：

```text
bootstrap_samples
```

新增：

```text
main_split_seed = 17
training_seeds = [7, 17, 29]
robustness_split_seeds = [23, 41, 59]
robustness_training_seed = 17
```

---

# 36. Horizon Strategy 重构

不再只使用：

```text
recursive_training = true / false
```

改为：

```text
horizon_strategy
```

支持：

```text
one_step
max_available
random_available
```

对应：

```text
Baseline      -> one_step
Recursive-Max -> max_available
RHRT          -> random_available
```

---

# 37. `data.py` 修改

Training dataset 支持：

```text
horizon_strategy
```

逻辑：

### one_step

\[
k=1
\]

### max_available

\[
k=K_t
\]

### random_available

\[
k\sim Uniform(1,K_t)
\]

其中：

\[
K_t=\min(3,T-t)
\]

只有 `random_available` 需要根据 epoch 重新采样。

同时记录：

```text
horizon_sampling_counts
```

---

# 38. `training.py` 修改

Variants：

```text
baseline
recursive_max
rhrt
ensemble
rhrt_ensemble
```

对应：

```text
baseline:
    horizon_strategy = one_step
    ensemble = false

recursive_max:
    horizon_strategy = max_available
    ensemble = false

rhrt:
    horizon_strategy = random_available
    ensemble = false

ensemble:
    horizon_strategy = one_step
    ensemble = true

rhrt_ensemble:
    horizon_strategy = random_available
    ensemble = true
```

---

# 39. `reporting.py` 修改

删除：

```text
bootstrap CI
paired bootstrap test
rho bootstrap
matched H3 slope
P(improvement)

rhrt_pass
ensemble_pass
stage1_pass
```

Prediction 部分只保留：

```text
MSE@1
MSE@2
MSE@3
Long MSE

mean
std

Relative Improvement
```

Secondary Prediction：

```text
Cos@1
Cos@2
Cos@3
```

Ensemble Reliability：

```text
rho@1
rho@2
rho@3
macro rho

high_low_ratio@1
high_low_ratio@2
high_low_ratio@3

selective risk
```

---

# 40. 输出目录

```text
outputs/
└── stage1/
    └── <encoder>/
        ├── seed_7/
        ├── seed_17/
        ├── seed_29/
        └── reports/
            ├── dataset_stats.json
            ├── training_summary.json
            ├── prediction_metrics.json
            ├── prediction_comparison.json
            ├── uncertainty_metrics.json
            ├── selective_risk.json
            ├── split_robustness.json
            └── stage1_summary.json
```

不保存 CSV。

多指标结果在终端中以表格形式打印。

---

# 41. 各结果文件内容

## `dataset_stats.json`

```text
train_patient_count
val_patient_count
test_patient_count

H1_window_count
H2_window_count
H3_window_count

train_timepoint_count
val_timepoint_count
test_timepoint_count
```

## `training_summary.json`

```text
training_seed
best_epoch
best_validation_score

H1_sampling_count
H2_sampling_count
H3_sampling_count
```

## `prediction_metrics.json`

保存五个 variants 的：

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

保存：

```text
baseline_vs_recursive_max
baseline_vs_rhrt
recursive_max_vs_rhrt
ensemble_vs_rhrt_ensemble
```

每个 comparison 包含：

```text
RI@1
RI@2
RI@3
RI_long
```

## `uncertainty_metrics.json`

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

按 horizon 保存：

```text
risk_100
risk_80
risk_reduction
```

## `split_robustness.json`

```text
split_seed

baseline_long_mse
recursive_max_long_mse
rhrt_long_mse

rhrt_relative_improvement
```

## `stage1_summary.json`

不保存 `stage1_pass`。

只保存：

```text
main_encoder
main_split_seed

baseline_long_mse
recursive_max_long_mse
rhrt_long_mse

rhrt_relative_improvement
rhrt_vs_recursive_max_improvement

macro_uncertainty_spearman
macro_high_low_ratio
risk_reduction_80
```

---

# 42. 正式实验执行顺序

## Phase A：Data Audit

首先生成：

```text
dataset_stats.json
```

确认：

- Patient split
- H1/H2/H3 sample count
- Encoder alignment
- Treatment alignment
- Delta time

## Phase B：BrainIAC Main Experiment

固定：

```text
split_seed = 17
```

运行：

```text
baseline × seeds 7,17,29
recursive_max × seeds 7,17,29
rhrt × seeds 7,17,29
ensemble × seeds 7,17,29
rhrt_ensemble × seeds 7,17,29
```

## Phase C：BrainIAC Split Robustness

运行：

```text
split_seed = 23
split_seed = 41
split_seed = 59
```

固定：

```text
training_seed = 17
```

只运行：

```text
baseline
recursive_max
rhrt
```

## Phase D：MRI-CORE Robustness

首先运行：

```text
baseline
rhrt
rhrt_ensemble
```

如果需要完整 ablation，再补：

```text
recursive_max
ensemble
```

---

# 43. 最终实验问题

1. Conventional one-step dynamics 在 recursive rollout 下表现如何？
2. RHRT 是否改善 multi-step patient state prediction？
3. Mixed random horizon 是否是 RHRT 合理设计的一部分？
4. RHRT improvement 是否对 training initialization 稳定？
5. RHRT improvement 是否依赖特定 patient split？
6. RHRT 是否能够在不同 frozen MRI representations 上表现出类似趋势？
7. Ensemble disagreement 是否能够识别 prediction error 较大的 trajectory？

---

# 44. 推荐 Contribution 表述

## Contribution 1：RHRT

> We introduce Random-Horizon Recursive Training (RHRT) for patient dynamics modeling. RHRT randomly samples rollout horizons during training and recursively feeds predicted latent states back into the dynamics model, exposing the model to its own accumulated prediction errors over trajectories of varying lengths.

## Contribution 2：Ensemble Reliability

> We further employ ensemble dynamics to estimate trajectory-level reliability through model disagreement, enabling unreliable imagined trajectories to be identified before downstream planning.

---

# 45. 最终 Stage 1 方法逻辑

不写成：

\[
	ext{Random Horizon}
+
	ext{Recursive Training}
=
	ext{Two Contributions}
\]

而统一写成：

\[
\boxed{
\text{Random-Horizon Recursive Training}
}
\]

其中 Random Horizon 与 Recursive Feedback 共同构成 RHRT。

`recursive_max` 仅作为：

\[
\boxed{\text{RHRT Ablation}}
\]

最终 Stage 1 形成：

\[
\boxed{
\text{RHRT}
\rightarrow
\text{Accurate Multi-Step Patient Dynamics}
}
\]

以及：

\[
\boxed{
\text{Ensemble Dynamics}
\rightarrow
\text{Trajectory Reliability Estimation}
}
\]

进一步支持后续：

\[
\boxed{
\text{Reliable Patient World Model}
\rightarrow
\text{Reliability-Aware Planning}
}
\]
