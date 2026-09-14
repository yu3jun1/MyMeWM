# CLARITY × HAUWM：最小迁移验证框架

本仓库用于验证：在 CLARITY 的纵向 MRI latent 上，Horizon Sampling（HS）能否改善长跨度状态预测，以及独立 dynamics ensemble 的分歧能否排序预测风险。MRI encoder 支持 MRI-CORE 与 BrainIAC，两者使用完全相同的患者划分、治疗动作、训练配置和评估流程。

这只是进入治疗方案闭环前的 Stage 1。它不生成治疗建议，也不能证明治疗的因果效应。

## 核心约束

- MRI-CORE 和 BrainIAC 都只使用原始预训练权重，并严格冻结。
- 不加载 CLARITY 训练后的 LoRA、adapter 或任何任务微调权重。
- BrainIAC 固定使用 `lora_r=0`；MRI-CORE 显式关闭 encoder/mask-decoder adapter。
- 抽取过程使用 `torch.inference_mode()`，全部 encoder 参数均为 `requires_grad=False`。
- 抽取配置会写入 `extraction_metadata.json`，随后进入 trajectory provenance 和训练 checkpoint。
- 不跨 encoder 比较绝对 latent MSE；只比较每个 encoder 内 HS/ensemble 相对自身 baseline 的改善。

## 实验设计

每个 encoder 独立运行四组消融：

| Variant | Horizon Sampling | Ensemble | 训练目标 |
|---|---:|---:|---|
| `baseline` | 否 | 否 | 一步预测 |
| `hs` | 是 | 否 | 随机采样 `1..K_max` |
| `ensemble` | 否 | 是 | 一步预测 |
| `hs_ensemble` | 是 | 是 | 随机采样 `1..K_max` |

HS 模型从当前 latent、完整真实治疗序列和时间间隔直接预测 `z[t+k]`。Recursive rollout 每次推进一步；ensemble 的每个 head 延续自己的 latent particle，以保留随 horizon 累积的 epistemic disagreement。

正式配置位于 `configs/stage1.json`：最大 horizon 为 5、ensemble size 为 5、患者级固定划分为 70%/15%/15%，默认运行 3 个随机种子。MRI-CORE 与 BrainIAC 合计为 `2 encoders × 4 variants × 3 seeds = 24` 次训练。

## 目录

```text
CLARITY_HAUWM_Minimal/
├── README.md
├── LICENSE
├── pyproject.toml
├── configs/
│   └── stage1.json
├── src/clarity_hauwm/
│   ├── brainiac_extract.py
│   ├── mri_core_extract.py
│   ├── clarity_adapter.py
│   ├── data.py
│   ├── model.py
│   ├── training.py
│   ├── evaluation.py
│   ├── ablation.py
│   ├── encoder_comparison.py
│   └── cli.py
└── tests/
```

所有生成产物统一写入 `/data/tanyuejun/CLARITY_HAUWM_Minimal/`，不放入代码仓库。仓库内的 `data/`、`outputs/` 和 checkpoint 路径仍由 Git 忽略。

## 1. 环境与数据

推荐复用现有 Python 3.10 环境：

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
MRI_CORE_CKPT=/home/tanyuejun/CLARITY/mri_foundation/pretrained_weights/MRI_CORE_vitb.pth
ARTIFACT_ROOT=/data/tanyuejun/CLARITY_HAUWM_Minimal
```

MRI-CORE 官方实现和权重说明见 [mazurowski-lab/mri_foundation](https://github.com/mazurowski-lab/mri_foundation)。当前本机已有 MRI-CORE 源码，但未发现 `MRI_CORE_vitb.pth`；开始抽取前必须按官方说明下载并放到上面的路径。框架不会用随机权重或普通 SAM 权重代替。

原始 MRI 数据中存在少量不完整 timepoint。两个抽取器都会跳过缺少任一模态的 timepoint；后续 encoder 对比要求最终患者和 timepoint 覆盖完全一致，否则拒绝训练。

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
  --checkpoint /home/tanyuejun/CLARITY/mri_foundation/pretrained_weights/MRI_CORE_vitb.pth \
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
  --output /data/tanyuejun/CLARITY_HAUWM_Minimal/trajectories/brainiac \
  --action-anchor source \
  --pooling mean

clarity-hauwm build-clarity \
  --timeline /home/tanyuejun/CLARITY/Predictor/dataset/MU_Glioma_Post/clinical_latest.json \
  --latents /data/tanyuejun/CLARITY_HAUWM_Minimal/latents/mri_core \
  --output /data/tanyuejun/CLARITY_HAUWM_Minimal/trajectories/mri_core \
  --action-anchor source \
  --pooling mean
```

`source` 表示 `A_t` 包含从当前 MRI 到下一次 MRI 之间、记录在源端及中间节点的治疗，不把目标 MRI 节点才记录的动作泄漏给模型。只有数据字典明确说明治疗属于目标区间时才使用 `destination`，并且两个 encoder 必须保持一致。

检查两套轨迹：

```bash
clarity-hauwm validate-data --data /data/tanyuejun/CLARITY_HAUWM_Minimal/trajectories/brainiac
clarity-hauwm validate-data --data /data/tanyuejun/CLARITY_HAUWM_Minimal/trajectories/mri_core
```

每位患者至少需要两个有效 MRI timepoint。模型按患者划分 train/validation/test，不会把同一患者的不同 timepoint 分到不同集合。

## 5. 运行 Encoder × HS/Ensemble 正式实验

```bash
clarity-hauwm compare-encoders \
  --encoder-data \
    brainiac=/data/tanyuejun/CLARITY_HAUWM_Minimal/trajectories/brainiac \
    mri_core=/data/tanyuejun/CLARITY_HAUWM_Minimal/trajectories/mri_core \
  --config configs/stage1.json \
  --output /data/tanyuejun/CLARITY_HAUWM_Minimal/outputs/encoder_comparison \
  --seeds 7 17 29 \
  --bootstrap-samples 2000
```

训练开始前会强制核对：

- patient 集合；
- 每位患者的 timepoint 序列；
- action vocabulary 和逐区间 multi-hot action；
- 每个区间的 `delta_days`。

任何一项不一致都会报错，不会在不对齐的数据上继续比较。

## 6. 输出与判据

总报告位于：

```text
/data/tanyuejun/CLARITY_HAUWM_Minimal/outputs/encoder_comparison/encoder_comparison.json
```

每个 encoder 还会生成：

```text
/data/tanyuejun/CLARITY_HAUWM_Minimal/outputs/encoder_comparison/<encoder>/
├── stage1_report.json
└── seed_<seed>/<variant>/
    ├── best.pt
    ├── history.json
    ├── evaluation.json
    ├── direct_records.csv
    └── rollout_records.csv
```

正式结论只看 held-out patients，并检查三个预注册判据：

1. `hs_ensemble` 的 `k >= 2` direct normalized-latent MSE 低于 `baseline`，患者级 paired bootstrap 的 95% CI 不跨 0；
2. `hs_ensemble` 的 recursive rollout MSE-horizon slope 低于 `baseline`；
3. ensemble uncertainty 与真实 squared error 的 Spearman 相关为正，且最高不确定性五分位的误差高于最低五分位。

`stage1_pass=true` 只有在三项同时通过时成立。若仅均值误差改善，只能说明 HS 可能有效，不能声称 uncertainty 已校准。

不同 encoder 的维度和 latent 几何不同，因此不能用绝对 MSE 判断 MRI-CORE 或 BrainIAC 谁更好。`encoder_comparison.json` 比较的是各 encoder 内 `hs_ensemble` 相对自身 baseline 的 direct error 改善比例、recursive slope 改善及 uncertainty ranking。

## 7. 结果边界

Stage 1 使用观测数据中的真实治疗序列，只验证 world-model rollout 和不确定性机制能否迁移到 CLARITY。即使通过，也不能直接进行逐步治疗推荐。下一阶段仍需单独设计候选动作生成、结局/utility 模型、off-policy 或因果评估、安全约束和临床审核。

代码测试：

```bash
/home/tanyuejun/miniconda3/envs/py310/bin/python -m pytest -q
```
