# Stage 1 重新设计：从“治疗推荐”退回可证伪的迁移实验

## 1. 研究问题

本阶段不证明某种治疗优于另一种治疗，而验证两个基础命题：

- `H1`：在相同容量和 patient split 下，训练时覆盖可变跨度，能降低 held-out 患者的长跨度 latent 预测误差；
- `H2`：独立初始化并使用 bootstrap 子样本训练的 dynamics heads，其 disagreement 能排序真实预测误差；
- `H3`：HS 与 ensemble 合用时，递归 rollout 的误差增长速度低于一步模型。

任何一个命题都可能失败。框架必须输出失败结论，而不是自动把 ensemble variance 解释为临床风险。

## 2. 与 HAUWM 的对应关系

HAUWM 从 `k ~ Uniform{1,...,K_max}` 采样未来跨度，使用独立 dynamics heads 预测未来 latent，并以 ensemble mean 重建、以 head disagreement 表示不确定性。论文还加入负的、按 `k` 线性放大的 HCU loss，使分歧随 horizon 增大。

迁移到 CLARITY 时有三处刻意变化：

1. 动力学必须条件化在真实治疗序列与不规则 `delta_days` 上；
2. MRI encoder 冻结，问题缩成低成本 latent regression；
3. 依据原 Stage 1 方案，HCU loss 暂不加入主实验。小样本医疗轨迹中直接最大化方差可能产生无界或虚假的“校准”。先验证自然 ensemble disagreement 是否与误差相关，再单独预注册 HCU/校准实验。

论文主体写 `M=5`，最终超参数表和附录 ensemble 消融报告 `M=7`；本项目默认 `M=5` 以遵循本地方案，并保持可配置。

## 3. 数据契约与防泄漏

每个患者一个轨迹文件：

```text
latents     float32 [T, D]
actions     float32 [T-1, A]  multi-hot
delta_days  float32 [T-1]
timepoints  unicode [T]
```

约束：`T >= 2`、时间严格前进、数组有限、所有动作处于 `[0,1]`。所有 MRI timepoint 必须来自同一个冻结 encoder。拆分按患者完成，latent mean/std 只在训练患者上拟合。

治疗对齐是最大的语义风险。默认 `source`：相邻可用 MRI 之间，合并源 timepoint 及中间 timepoint 的动作，但不包含目标 timepoint；这避免 target-side action leakage。最终实验前必须用原始数据字典确认记录语义。

## 4. 最小模型

```text
actions[0:k] + log(delta_days) -> Linear + GRU -> c_action
k embedding + log(total_days)                 -> c_horizon
[z_t, c_action, c_horizon] -> MLP head_m -> residual_m
z_hat_m = z_t + residual_m
prediction = mean_m(z_hat_m)
uncertainty = mean_D Var_m(z_hat_m)
```

每个 head 独立初始化。训练 ensemble 时，每个 batch 为每个 head 采样独立 bootstrap mask；这是为有限医疗数据保留 epistemic diversity 的工程化补充。共享 action/horizon encoder，只有 dynamics heads 独立，控制参数规模。

损失为每个 head 的 `MSE + alpha * cosine_distance`。Stage 1 不含 survival、policy、counterfactual ranking、HCU 或 clinical utility loss。

## 5. 两种评估

- Direct：固定真实 `z_t`，一次输入全部 GT action/delta 序列，直接预测 `z_{t+k}`。
- Recursive：从真实 `z_0` 开始逐步推进。ensemble 的第 `m` 个 head 延续自己的预测粒子，而不是每步把 ensemble mean 重新喂给所有 heads。

报告 normalized-latent MSE、cosine distance、uncertainty、error-vs-uncertainty Spearman、上下 uncertainty 五分位误差比及 MSE-horizon 线性斜率。统计单位是患者；paired bootstrap 先在患者内平均，再对患者重采样，避免把同一患者的多个窗口误当独立样本。

## 6. Encoder 对照轴

BrainIAC 与 MRI-CORE 必须分别抽取、分别标准化、分别完成四组消融。只比较每个 encoder 内 `HS+ensemble` 相对自身 baseline 的改善和不确定性排序；由于 latent 维度及几何不同，不比较跨 encoder 的绝对 latent MSE。`compare-encoders` 在训练前强制核对 patient、timepoint、action 与时间间隔完全对齐。

## 7. 最小闭环的后续接口（不在本阶段实现）

Stage 1 通过后，下一阶段才把 GT action 替换为候选治疗序列：

```text
当前真实 MRI latent
    -> 规则约束生成少量候选治疗块
    -> ensemble rollout 每个候选
    -> 预测结局模型给 utility，disagreement 给 uncertainty penalty
    -> 选择或拒绝（abstain）
    -> 等待下一次真实 MRI/结局
    -> 写回 trajectory，监测 calibration drift，再训练
```

第一版闭环必须只做离线 replay：在每个历史决策点隐藏后续记录，候选集中强制包含真实治疗，禁止把未观测反事实当 ground truth。只有 dynamics、outcome calibration、positivity/overlap 和 safety guardrail 分别通过后，才能讨论前瞻性临床验证。

