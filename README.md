# CLARITY × HAUWM：最小迁移验证框架

这个仓库只回答一个问题：在冻结的 CLARITY MRI latent 上，`Horizon Sampling (HS)` 是否改善长跨度预测，独立动力学 ensemble 的分歧是否能排序预测风险。

它不是治疗推荐器，也不输出临床建议。Stage 1 全程使用数据中的真实治疗序列（ground-truth actions），不训练 Policy、Survival 或 LLM Agent。只有三个预注册判据通过后，才值得进入候选治疗闭环。

## 最小实验

四个模型使用完全相同的数据划分、latent 标准化、网络容量和优化器：

| 名称 | 随机 horizon 训练 | ensemble | 训练跨度 |
|---|---:|---:|---|
| `baseline` | 否 | 否 | 仅一步 |
| `hs` | 是 | 否 | `1..K_max` |
| `ensemble` | 否 | 是 | 仅一步 |
| `hs_ensemble` | 是 | 是 | `1..K_max` |

HS 模型直接编码完整的真实治疗/时间间隔序列，并预测 `z[t+k]`。递归 rollout 每次只推进一步；ensemble rollout 会让每个 head 延续自己的 latent particle，因而保留随 rollout 累积的 epistemic disagreement。

## 快速验证代码

推荐复用已有 Python 3.10 环境：

```bash
cd /home/tanyuejun/CLARITY_HAUWM_Minimal
/home/tanyuejun/miniconda3/envs/py310/bin/python -m pip install -e . --no-deps

clarity-hauwm synthesize --output data/synthetic --patients 80 --seed 7
clarity-hauwm validate-data --data data/synthetic
clarity-hauwm ablate \
  --data data/synthetic \
  --config configs/smoke.json \
  --output outputs/smoke \
  --seeds 7
```

合成数据只用于检查数据流、训练、四组消融、direct/recursive 评估和报告是否工作，不能作为迁移有效性的证据。

## 准备真实 CLARITY 数据

### 1. 一次性抽取冻结 BrainIAC latent

脚本会跳过已有 `.npy`，可断点续跑。默认输出每个 timepoint 的 32 个 token 的均值，即 768 维向量。

```bash
clarity-hauwm extract-brainiac \
  --clarity-root /home/tanyuejun/CLARITY \
  --timeline /home/tanyuejun/CLARITY/Predictor/dataset/MU_Glioma_Post/clinical_latest.json \
  --mri-root /data/tanyuejun/CLARITY/dataset/MU-Glioma-Post \
  --brainiac-checkpoint /home/tanyuejun/CLARITY/BrainIAC-main/src/checkpoints/BrainIAC.ckpt \
  --clarity-checkpoint /home/tanyuejun/CLARITY/experiments/exp012_cf_diversity/checkpoints/best_loss.pth \
  --output data/brainiac_latents \
  --device cuda
```

若要严格冻结原始 BrainIAC（不加载 CLARITY 训练后的 LoRA），删除 `--clarity-checkpoint`。

### 2. 对齐时间线与 latent

```bash
clarity-hauwm build-clarity \
  --timeline /home/tanyuejun/CLARITY/Predictor/dataset/MU_Glioma_Post/clinical_latest.json \
  --latents data/brainiac_latents \
  --output data/clarity_trajectories \
  --action-anchor source

clarity-hauwm validate-data --data data/clarity_trajectories
```

`source` 明确定义 `A_t` 为从本次 MRI 到下一次 MRI 之间、在源端及其中间记录的治疗；不会把目标 MRI 处才记录的动作泄漏到输入。若数据字典最终确认治疗记录属于结束区间，必须把该参数改为 `destination`，并在所有消融中保持一致。

### 3. 运行正式消融

```bash
clarity-hauwm ablate \
  --data data/clarity_trajectories \
  --config configs/stage1.json \
  --output outputs/clarity_stage1 \
  --seeds 7 17 29
```

主要产物：

- 每个 run 的 `best.pt`、`history.json`；
- `direct_records.csv` 与 `rollout_records.csv`；
- 按 horizon 汇总的 `evaluation.json`；
- 四组聚合的 `stage1_report.json`，包含 patient-level paired bootstrap 置信区间和是否达到预注册标准。

## 通过标准

正式结论只看 held-out patient，不能随机拆 timepoint：

1. `hs_ensemble` 在 `k >= 2` 的 direct MSE 低于 `baseline`，patient-level paired bootstrap 的 95% CI 不跨 0；
2. `hs_ensemble` 的 recursive rollout MSE-horizon 斜率低于 `baseline`；
3. ensemble uncertainty 与真实 squared error 的 Spearman 相关为正，且高不确定性五分位的平均误差高于低五分位。

若只改善均值误差、但不确定性不排序误差，只能证明 HS 有效，不能声称 uncertainty 已校准。完整设计、风险点和下一阶段闭环接口见 [DESIGN.md](DESIGN.md)。

