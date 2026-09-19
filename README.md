# CLARITY × RRT × Ensemble：Stage 1

本仓库验证两个问题：Recursive Rollout Training（RRT）能否改善多步 MRI latent 预测；独立 dynamics ensemble 的分歧能否识别高误差轨迹。主实验使用冻结的 BrainIAC encoder，MRI-CORE 用于 representation robustness。两个 latent space 只比较各自内部的相对改善，不直接比较绝对 MSE。

四个 variant 为 `baseline`、`rrt`、`ensemble`、`rrt_ensemble`。Baseline 和 Ensemble 使用一步训练；RRT 在每个起点使用不超过 `Kmax=3` 的最大可用 horizon，递归回灌模型预测的 latent，只对终点计算 MSE。两类方法共享相同的单步 transition architecture，输入均为 `(latent_state, treatment, delta_time)`，没有 target-horizon conditioning 或 teacher forcing。Ensemble 的 5 个 member 独立初始化和打乱 batch，各自维持自己的 rollout 状态。所有方法用 validation MSE@2 与 MSE@3 的均值选择 checkpoint。

主实验在患者级 70%/15%/15% 划分上评价 H1–H3，split seed 为 17，training seeds 为 7、17、29。主指标是 MSE@1、MSE@2、MSE@3 和 Long MSE = (MSE@2 + MSE@3)/2；Cos@1–3 作为辅助指标。报告跨 seed 的 mean ± std，并按同一 seed 计算 Baseline→RRT 与 Ensemble→RRT+Ensemble 的相对改善。患者划分稳健性使用 split seeds 23、41、59，固定 training seed 17。Reliability 分析使用逐 horizon Spearman、分歧最高/最低三分位误差比及 selective risk。Ensemble disagreement 只是 prediction risk signal，不解释为校准后的不确定性。程序只报告实验事实，不做自动 Pass/Fail 判断。

## 1. 环境与数据

推荐复用现有 Python 3.10 环境。本机运行实验时默认只在 GPU 4–7 中按实时负载选卡，除非用户明确指定，否则避免使用 GPU 0–3：

```bash
cd /home/tanyuejun/CLARITY_HAUWM_Minimal
/home/tanyuejun/miniconda3/envs/py310/bin/python -m pip install -e ".[mri,dev]"
```

实验使用以下路径：

```text
CLARITY_ROOT=/home/tanyuejun/CLARITY
TIMELINE=/home/tanyuejun/CLARITY/Predictor/dataset/MU_Glioma_Post/clinical_latest.json
MRI_ROOT=/data/tanyuejun/CLARITY/dataset/MU-Glioma-Post
BRAINIAC_CKPT=/home/tanyuejun/CLARITY/BrainIAC-main/src/checkpoints/BrainIAC.ckpt
MRI_CORE_ROOT=/home/tanyuejun/CLARITY/mri_foundation
MRI_CORE_CKPT=/home/tanyuejun/CLARITY/mri_foundation/pretrained_weights/mri_foundation.pth
SAM_CKPT="/home/tanyuejun/CLARITY/mri_foundation/SAM weights/sam_vit_b_01ec64.pth"
LATENT_ROOT=/data/tanyuejun/CLARITY_HAUWM_Minimal/latents
LARGE_ARTIFACT_ROOT=/data/tanyuejun/CLARITY_HAUWM_Minimal/large_artifacts
```

MRI-CORE 官方实现和权重说明见 [mazurowski-lab/mri_foundation](https://github.com/mazurowski-lab/mri_foundation)。本机的 `mri_foundation.pth` 提供 MRI 预训练 ViT backbone；`sam_vit_b_01ec64.pth` 只补齐 MRI checkpoint 未包含的 SAM neck。加载时严格核对 neck 的 6 个张量，随后冻结整个 `image_encoder`。框架不使用随机 neck、mask decoder 或任何 CLARITY 微调权重。

原始 MRI 数据中存在少量不完整 timepoint。两个抽取器都会跳过缺少任一模态的 timepoint；运行双 encoder 稳健性实验前应使用 validate-encoder-alignment 核对最终患者和 timepoint 覆盖。

为避免误复用历史 LoRA latent，两个抽取命令都要求 `--output` 是空目录；目录中存在任何文件时会立即报错。

## 2. 抽取严格冻结的 BrainIAC latent

```bash
clarity-hauwm extract-brainiac \
  --clarity-root /home/tanyuejun/CLARITY \
  --timeline /home/tanyuejun/CLARITY/Predictor/dataset/MU_Glioma_Post/clinical_latest.json \
  --mri-root /data/tanyuejun/CLARITY/dataset/MU-Glioma-Post \
  --brainiac-checkpoint /home/tanyuejun/CLARITY/BrainIAC-main/src/checkpoints/BrainIAC.ckpt \
  --output /data/tanyuejun/CLARITY_HAUWM_Minimal/latents/brainiac \
  --device cuda \
  --tokens-per-modality 8 \
  --output-kind mean
```

输出为每个 timepoint 一个 768 维向量。CLI 不提供 CLARITY checkpoint 或 LoRA 参数，因此不能意外加载任务微调后的 encoder。

## 3. 抽取严格冻结的 MRI-CORE latent

```bash
clarity-hauwm extract-mri-core \
  --mri-core-root /home/tanyuejun/CLARITY/mri_foundation \
  --timeline /home/tanyuejun/CLARITY/Predictor/dataset/MU_Glioma_Post/clinical_latest.json \
  --mri-root /data/tanyuejun/CLARITY/dataset/MU-Glioma-Post \
  --checkpoint /home/tanyuejun/CLARITY/mri_foundation/pretrained_weights/mri_foundation.pth \
  --sam-checkpoint "/home/tanyuejun/CLARITY/mri_foundation/SAM weights/sam_vit_b_01ec64.pth" \
  --output /data/tanyuejun/CLARITY_HAUWM_Minimal/latents/mri_core \
  --device cuda \
  --image-size 1024 \
  --normalization minmax \
  --slice-policy all \
  --slice-batch-size 2 \
  --output-kind mean
```

MRI-CORE 是 2D encoder。每个 axial slice 独立归一化到 `[0,1]`，复制为三通道并送入冻结的 ViT-B `image_encoder`；feature map 经空间平均得到 256 维 slice token，再对四个模态和全部切片求均值得到 timepoint latent。这与官方特征抽取输入规范一致。

如果现有 CLARITY MRI-CORE latent 使用了不同的切片选择、resize 或 normalization，必须用其原始参数重新抽取两边数据，不能把预处理差异归因于 encoder。可选的 `--normalization sam` 会额外施加 ImageNet/SAM mean-std；正式实验应预先固定一种设置。

显存不足时可以降低 `--slice-batch-size`。如果预注册只使用固定数量切片，可改为：

```bash
--slice-policy uniform --slices-per-modality 16
```

正式报告必须记录这一变化，并保证所有 MRI-CORE run 完全一致。

## 4. 构建 CLARITY 轨迹

分别把两个 encoder 的 latent 与同一份临床时间线对齐：

```bash
clarity-hauwm build-clarity \
  --timeline /home/tanyuejun/CLARITY/Predictor/dataset/MU_Glioma_Post/clinical_latest.json \
  --latents /data/tanyuejun/CLARITY_HAUWM_Minimal/latents/brainiac \
  --output data/trajectories/brainiac \
  --action-anchor source \
  --pooling mean

clarity-hauwm build-clarity \
  --timeline /home/tanyuejun/CLARITY/Predictor/dataset/MU_Glioma_Post/clinical_latest.json \
  --latents /data/tanyuejun/CLARITY_HAUWM_Minimal/latents/mri_core \
  --output data/trajectories/mri_core \
  --action-anchor source \
  --pooling mean
```

`source` 表示 `A_t` 包含从当前 MRI 到下一次 MRI 之间、记录在源端及中间节点的治疗，不把目标 MRI 节点才记录的动作泄漏给模型。只有数据字典明确说明治疗属于目标区间时才使用 `destination`，并且两个 encoder 必须保持一致。

检查两套轨迹：

```bash
clarity-hauwm validate-data --data data/trajectories/brainiac
clarity-hauwm validate-data --data data/trajectories/mri_core
```

每位患者至少需要两个有效 MRI timepoint。模型按患者划分 train/validation/test，不会把同一患者的不同 timepoint 分到不同集合。

## 5. 运行 Stage 1

先核对两套 encoder 的患者、timepoint、action 和 delta_days：

```bash
clarity-hauwm validate-encoder-alignment --encoder-data brainiac=data/trajectories/brainiac mri_core=data/trajectories/mri_core
```

BrainIAC 是主实验。先做数据审计，再训练和评估四个 variant：

```bash
clarity-hauwm audit-stage1 --data data/trajectories/brainiac --config configs/stage1_recursive.json --output outputs/stage1/brainiac
clarity-hauwm train-stage1 --data data/trajectories/brainiac --config configs/stage1_recursive.json --output outputs/stage1/brainiac
clarity-hauwm evaluate-recursive --input outputs/stage1/brainiac
clarity-hauwm evaluate-uncertainty --input outputs/stage1/brainiac
clarity-hauwm train-split-robustness --data data/trajectories/brainiac --config configs/stage1_recursive.json --output outputs/stage1/brainiac
clarity-hauwm train-horizon-ablation --data data/trajectories/brainiac --config configs/stage1_recursive.json --output outputs/stage1/brainiac
clarity-hauwm summarize-stage1 --input outputs/stage1/brainiac
```

`train-stage1` 默认运行四个 variant × seeds 7、17、29，可以通过 `--variants` 和 `--seeds` 分批运行。`train-split-robustness` 在三个新增 patient split 上仅运行 `baseline` 与 `rrt`，固定 training seed 17，自动评估 H1–H3。`train-horizon-ablation` 在主 split 上固定 training seed 17，分别训练 Kmax=1、2、3，并使用相同的 H1–H3 测试窗口；K1 使用一步 Baseline，K2/K3 使用 RRT。RRT 的 K2/K3 消融在每个起点使用该上限内的最大可用训练 horizon；Baseline 固定一步。

数据审计也列出 H4/H5 在 train、validation、test 中的窗口数。仅在每个 split 的 H4 和 H5 窗口都达到阈值时，可显式运行 Kmax=5 压力测试：

```bash
clarity-hauwm train-horizon-ablation --data data/trajectories/brainiac --config configs/stage1_recursive.json --output outputs/stage1/brainiac --include-stress --min-stress-windows 30
```

这里的阈值默认为每个 split、每个 horizon 至少 30 个窗口，可按研究设计调整。K5 的 H4/H5 指标仅作描述；主实验和 Long MSE 仍限 H1–H3。

MRI-CORE 使用独立 dynamics 模型做 representation robustness，先运行以下三个 variant：

```bash
clarity-hauwm audit-stage1 --data data/trajectories/mri_core --config configs/stage1_recursive.json --output outputs/stage1/mri_core
clarity-hauwm train-stage1 --data data/trajectories/mri_core --config configs/stage1_recursive.json --variants baseline rrt rrt_ensemble --output outputs/stage1/mri_core
clarity-hauwm evaluate-recursive --input outputs/stage1/mri_core
clarity-hauwm evaluate-uncertainty --input outputs/stage1/mri_core
clarity-hauwm summarize-stage1 --input outputs/stage1/mri_core
```

每次主实验 run 保存在 `seed_<training_seed>/<variant>/`，训练 horizon 消融保存在 `horizon_ablation/k1`、`k2`、`k3`（可选 `k5`）。汇总写入 `reports/`：`dataset_stats.json`、`training_summary.json`、`prediction_metrics.json`、`prediction_comparison.json`、`uncertainty_metrics.json`、`selective_risk.json`、`split_robustness.json`、`horizon_ablation.json` 和 `stage1_summary.json`。终端同步打印结果表格，不生成 CSV。训练 horizon 计数按每个 member 的唯一训练起点记录，不乘以 epoch 数。

MSE@k 是所有合法测试窗口的平均 normalized latent MSE；每个 seed 独立计算，跨 seed 使用样本标准差。相对改善按同一 seed 配对计算，再汇总 mean ± std，单位为百分比。Spearman 和最高/最低三分位误差比只在同一个 horizon 内计算。Selective risk 删除 ensemble disagreement 最高的 20% 窗口，对其余窗口计算平均 MSE；Risk Reduction 的单位为百分比。缺少有效数据时相应 JSON 值为 `null`。已有旧协议 checkpoint 不能用于新版模型，需要重新训练。

代码测试：

```bash
/home/tanyuejun/miniconda3/envs/py310/bin/python -m pytest -q
```

## 6. MRI-CORE LoRA 补充实验

此补充实验按 [`stage1_mricore_lora_experiment_plan.md`](stage1_mricore_lora_experiment_plan.md) 运行。LoRA 适配只用主 split（seed 17）的 train 患者更新 Q/V 低秩参数和独立的一步 dynamics adapter；validation 患者只用于选择 `val_one_step_mse` 最低的 checkpoint，test 患者只用于最终评估。原 MRI-CORE 参数冻结。LoRA 默认 rank 8、alpha 16、dropout 0.05、anchor 权重 0.1，配置在 `configs/mri_core_lora.json`。当前正式运行中，冻结主干使用 BF16 计算、可训练 LoRA 与 Dynamics 参数保留 FP32；每次前向/反向使用 16 张切片，3 张 GPU 以 CPU Gloo 汇总全局 batch 32 的梯度。配置记录精度；checkpoint 和特征 provenance 另外记录参与适配的 GPU 进程数。

在仓库根目录依次运行（按实际可用 GPU 修改 `CUDA_VISIBLE_DEVICES`）：

```bash
CUDA_VISIBLE_DEVICES=3,6,7 OMP_NUM_THREADS=4 torchrun --standalone --nnodes=1 --nproc-per-node=3 --no-python /home/tanyuejun/miniconda3/envs/py310/bin/clarity-hauwm adapt-mri-core-lora
CUDA_VISIBLE_DEVICES=7 clarity-hauwm extract-mri-core-lora --resume
CUDA_VISIBLE_DEVICES=7 clarity-hauwm train-mri-core-lora-dynamics
clarity-hauwm summarize-mri-core-lora
```

`bash scripts/run_mri_core_lora_full.sh` 可从空目录独立执行整套实验；它在适配完成后调用 `scripts/continue_mri_core_lora_experiment.sh --completed` 自动执行后三步。启动和进度分别写入 `logs/pipeline_bootstrap.log` 与 `logs/pipeline.log`，适配器启动输出写入 `logs/launcher.log`。

重抽取若被中断，使用 `clarity-hauwm extract-mri-core-lora --resume` 继续。Dynamics 阶段再次执行时会校验并跳过已完整训练的 seed run，再统一评估。要重新做 LoRA 适配，使用新的 `mri_core_lora*` 输出根目录，并为后续命令传入相同的 `--output`；已有 checkpoint 不会被覆盖。

所有新增产物独立放在 `outputs/stage1/mri_core_lora/`，与 `outputs/stage1/mri_core/` 的冻结实验区分：

```text
outputs/stage1/mri_core_lora/
├── adaptation/                # best_lora.pt、config.json、training_log.json、adaptation_summary.*
├── features/
│   ├── latents/               # 重新抽取的每时间点 256 维 latent
│   ├── trajectories/          # 与原 MRI-CORE 对齐的 Stage 1 数据集
│   ├── train.pt、validation.pt、test.pt
│   ├── normalization.json     # 仅按 train 患者重新拟合
│   └── provenance.json
├── seed_7|seed_17|seed_29/
│   └── baseline|rrt/          # 独立的 Dynamics checkpoint、训练和评估结果
├── logs/                      # adaptation.log、extraction.log、dynamics.log、reporting.log
└── reports/
    ├── prediction_metrics.json、prediction_comparison.json   # LoRA 内部比较
    ├── four_group_prediction_metrics.json                     # Frozen/LoRA × Baseline/RRT
    ├── paired_relative_improvement.json                       # 按 seed 配对 RI
    ├── patient_macro_metrics.json                             # 患者等权误差
    ├── representation_diagnostics.json                        # drift 与 smoothness
    ├── comparison_provenance.json
    └── summary.md
```

当前冻结协议每时间点使用四个模态的全部切片（示例 MRI 共 620 张），LoRA 适配需要逐切片反向传播，计算开销较高。特征重抽取复用冻结 MRI-CORE 的模态、切片、空间预处理与 mean pooling；提取后验证患者、时间点、action 和间隔对齐。Dynamics 复用 `configs/stage1_recursive.json`，仅训练 Baseline 和 RRT，各运行 seed 7、17、29。主比较是 **LoRA Baseline 与 LoRA RRT** 在同一 representation 下的 H1–H3 和 Long MSE；跨 Frozen/LoRA 的绝对 MSE 仅作描述，因为 latent 几何可能改变。全部四组结果与患者等权指标在 `summary.md` 和对应 JSON 中分别保存。
