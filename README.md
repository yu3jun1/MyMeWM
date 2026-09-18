# CLARITY × RHRT × Ensemble：Stage 1

本仓库验证两个问题：随机 horizon 的递归训练能否减少多步 MRI latent 预测误差；独立 dynamics ensemble 的分歧能否识别高误差轨迹。主实验使用冻结的 BrainIAC encoder，MRI-CORE 采用相同协议作为稳健性实验。两个 latent space 只比较各自内部的相对改善，不比较绝对 MSE。

五个 variant 是 baseline、recursive_max、rhrt、ensemble、rhrt_ensemble。Baseline 和 Ensemble 用一步训练；Recursive-Max 在每个起点使用最大可用 horizon；RHRT 在 1 到 min(3, 剩余步数) 中均匀采样，递归回灌预测 latent，只对终点计算 loss。Recursive-Max 用来检验随机 horizon 的作用，不作为独立 contribution。Ensemble 的 5 个 member 独立初始化、打乱 batch 和采样 horizon，且各自维护 rollout 状态。所有方法均以 validation MSE@2 和 MSE@3 的平均值选择 checkpoint。

正式比较限定 horizon 1、2、3。主实验以患者级 70%/15%/15% 划分，split seed 为 17；training seeds 为 7、17、29。主指标是 MSE@1、MSE@2、MSE@3 和 Long MSE = (MSE@2 + MSE@3)/2，报告跨 seed 的 mean ± std 及相对改善百分比。患者划分稳健性使用 split seeds 23、41、59，并固定 training seed 17。风险信号使用逐 horizon Spearman、最高/最低 uncertainty 三分位误差比和 selective risk。程序只报告结果，不给实验自动判定。Stage 1 只评价观测治疗序列下的预测与风险排序。

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

先确认两套 encoder 的患者、timepoint、action 和 delta_days 对齐：

```bash
clarity-hauwm validate-encoder-alignment --encoder-data brainiac=data/trajectories/brainiac mri_core=data/trajectories/mri_core
```

BrainIAC 是主实验。使用 GPU 时，先根据实时负载把 `CUDA_VISIBLE_DEVICES` 设为 4–7 中的一张卡。先运行数据审计，核对患者数、timepoint 数和各 horizon 的合法窗口数：

```bash
clarity-hauwm audit-stage1 --data data/trajectories/brainiac --config configs/stage1_recursive.json --output outputs/stage1/brainiac
clarity-hauwm train-stage1 --data data/trajectories/brainiac --config configs/stage1_recursive.json --output outputs/stage1/brainiac
clarity-hauwm evaluate-recursive --input outputs/stage1/brainiac
clarity-hauwm evaluate-uncertainty --input outputs/stage1/brainiac
clarity-hauwm train-split-robustness --data data/trajectories/brainiac --config configs/stage1_recursive.json --output outputs/stage1/brainiac
clarity-hauwm summarize-stage1 --input outputs/stage1/brainiac
```

`train-stage1` 默认运行五个 variant × seeds 7、17、29。`train-split-robustness` 运行三个新增 patient split，各运行 baseline、recursive_max、rhrt × seed 17，并自动评估。长时间实验可以用 `--variants` 和 `--seeds` 分批运行主实验。完成后再运行汇总。

MRI-CORE 作为 representation robustness，先运行 baseline、rhrt、rhrt_ensemble。它使用自己的 dynamics 模型，不与 BrainIAC 比绝对 MSE：

```bash
clarity-hauwm audit-stage1 --data data/trajectories/mri_core --config configs/stage1_recursive.json --output outputs/stage1/mri_core
clarity-hauwm train-stage1 --data data/trajectories/mri_core --config configs/stage1_recursive.json --variants baseline rhrt rhrt_ensemble --output outputs/stage1/mri_core
clarity-hauwm evaluate-recursive --input outputs/stage1/mri_core
clarity-hauwm evaluate-uncertainty --input outputs/stage1/mri_core
clarity-hauwm summarize-stage1 --input outputs/stage1/mri_core
```

每次 run 保存在 `seed_<training_seed>/<variant>/`，包含 checkpoint、训练记录和逐窗口评估结果。汇总写入 `reports/`：`dataset_stats.json`、`training_summary.json`、`prediction_metrics.json`、`prediction_comparison.json`、`uncertainty_metrics.json`、`selective_risk.json`、`split_robustness.json` 和 `stage1_summary.json`。终端同时打印表格；不生成 CSV。

MSE@k 是该 horizon 所有合法测试窗口的平均误差；每个 seed 独立计算，跨 seed 使用样本标准差。相对改善按同一 seed 的两种方法计算，再汇总 mean ± std，单位为百分比。Spearman 和最高/最低三分位误差比在每个 horizon 的预测窗口内计算。Selective risk 删除 uncertainty 最高的 20% 窗口，对剩余窗口计算平均 MSE；Risk Reduction 的单位也是百分比。缺少有效数据时相应 JSON 值为 `null`。

旧 checkpoint 属于上一版实验协议，需要重新训练。代码测试：

```bash
/home/tanyuejun/miniconda3/envs/py310/bin/python -m pytest -q
```
