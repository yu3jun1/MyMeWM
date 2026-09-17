# CLARITY × RHRT × Ensemble：Stage 1

本仓库验证两个问题：随机 horizon 的递归训练能否减少多步 MRI latent 预测误差；独立 dynamics ensemble 的分歧能否识别高误差轨迹。主实验使用冻结的 BrainIAC encoder，MRI-CORE 采用相同协议作为稳健性实验。两个 latent space 只比较各自内部的相对改善，不比较绝对 MSE。

四个 variant 是 baseline、rhrt、ensemble、rhrt_ensemble。Baseline 用真实起始 latent 进行一步训练。RHRT 在每个合法起点均匀采样 1 到 min(3, 剩余步数)，逐步把预测状态送回模型，并只对最终状态计算标准化 latent MSE。Ensemble 的 5 个 member 各自初始化、采样、打乱 batch、训练并维护自己的 rollout 状态；不做 patient bootstrap。所有模型使用相同的 validation recursive MSE@2/@3 平均值选 checkpoint。

正式比较限定 horizon 1、2、3。患者划分固定为 70%/15%/15%，split seed 为 17；实验 seed 为 7、17、29。主判据为 baseline 对 rhrt 的 long-horizon MSE 与 matched H3 slope 改善，以及 rhrt_ensemble 的 within-horizon uncertainty-error Spearman 与患者级 bootstrap CI。Stage 1 只评价观测治疗序列下的预测与风险排序，不给出治疗建议。

训练配置在 configs/stage1_recursive.json。旧 Stage 1 checkpoint 使用不同目标和模型结构，必须重新训练。

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

建议先运行 BrainIAC，再用同样命令和配置运行 MRI-CORE。若两套轨迹都可用，先确认 patient、timepoint、action 与 delta_days 对齐：

    clarity-hauwm validate-encoder-alignment --encoder-data brainiac=data/trajectories/brainiac mri_core=data/trajectories/mri_core

BrainIAC 主实验。使用 GPU 时，先根据实时负载把 CUDA_VISIBLE_DEVICES 设为 4–7 中的一张卡：

    clarity-hauwm train-stage1 --data data/trajectories/brainiac --config configs/stage1_recursive.json --variants baseline rhrt ensemble rhrt_ensemble --seeds 7 17 29 --output outputs/stage1/brainiac
    clarity-hauwm evaluate-recursive --input outputs/stage1/brainiac --max-horizon 3
    clarity-hauwm evaluate-uncertainty --input outputs/stage1/brainiac --max-horizon 3
    clarity-hauwm summarize-stage1 --input outputs/stage1/brainiac --bootstrap-samples 2000

MRI-CORE 稳健性实验：

    clarity-hauwm train-stage1 --data data/trajectories/mri_core --config configs/stage1_recursive.json --variants baseline rhrt ensemble rhrt_ensemble --seeds 7 17 29 --output outputs/stage1/mri_core
    clarity-hauwm evaluate-recursive --input outputs/stage1/mri_core --max-horizon 3
    clarity-hauwm evaluate-uncertainty --input outputs/stage1/mri_core --max-horizon 3
    clarity-hauwm summarize-stage1 --input outputs/stage1/mri_core --bootstrap-samples 2000

每个 seed/variant 目录含 best.pt、training.json 和 recursive_metrics.json；ensemble variant 还含 uncertainty_metrics.json。递归与不确定性评估各自保存记录级 JSON，供患者级 bootstrap 使用，不生成 CSV。reports 目录分开保存 rhrt_summary.json、rhrt_bootstrap.json、ensemble_summary.json、ensemble_bootstrap.json 和精简的 stage1_summary.json。终端表格同时写入 reports 下的 .log 文件。

单次 recursive_metrics.json 的 MSE@k 是该 horizon 全部合法窗口的均值。跨 seed 的正式比较先对每位患者聚合，再平均患者指标；long-horizon MSE 是 MSE@2 与 MSE@3 的平均值。H3 slope 仅使用同一患者中可以完整展开三步的起点，先拟合患者 slope 再汇总。Spearman 在每个 horizon 内基于患者平均 uncertainty 与 error 计算，再跨 horizon 取宏平均；患者级 bootstrap 同时复采样所有实验 seed 的相同患者。

RQ1 的通过标准是 rhrt 的 long-horizon MSE 更低，改善量的 95% paired-bootstrap CI 下界大于零，并且 matched H3 slope 下降、改善量为正，bootstrap 改善概率大于 0.5。RQ2 使用 rhrt_ensemble：macro Spearman 大于零、bootstrap CI 下界大于零，且最高/最低 uncertainty 三分位误差比的宏平均大于一。该指标只支持风险排序，不代表概率校准。正式通过判定还要求四个 variant 均完成 seed 7、17、29，ensemble 各有 5 个 member，并使用固定患者划分；探索性少量运行仍输出指标，但不会标记通过。若数据不足以计算某指标，JSON 写 null，判据不通过。

代码测试：

    /home/tanyuejun/miniconda3/envs/py310/bin/python -m pytest -q
