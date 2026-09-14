# MRI-CORE 可选编码器与公平对比

MRI-CORE 官方实现是基于 SAM ViT-B 的 2D MRI encoder。体数据需切成 2D 图像；每个切片送入 `image_encoder` 后得到 `[256,H',W']` feature map，本框架对空间维平均为一个 256 维 slice token，再在 timepoint 层面对切片和四种模态聚合。

## 1. 权重准备

本地 `/home/tanyuejun/CLARITY/mri_foundation` 已包含代码，但审计时没有发现 `MRI_CORE_vitb.pth`。请按官方仓库说明下载 checkpoint，并显式传入路径；框架不会静默使用随机权重或普通 SAM 权重代替。

## 2. 抽取 MRI-CORE latent

默认按 MRI-CORE 官方特征抽取规范，对 axial slice 做 slice-wise min-max 到 `[0,1]`。若要与既有 CLARITY latent 严格对照，必须使用生成该批 latent 时完全相同的切片、resize 与 normalization 参数：

```bash
clarity-hauwm extract-mri-core \
  --mri-core-root /home/tanyuejun/CLARITY/mri_foundation \
  --timeline /home/tanyuejun/CLARITY/Predictor/dataset/MU_Glioma_Post/clinical_latest.json \
  --mri-root /data/tanyuejun/CLARITY/dataset/MU-Glioma-Post \
  --checkpoint /path/to/MRI_CORE_vitb.pth \
  --output data/mri_core_latents \
  --device cuda \
  --normalization minmax \
  --slice-policy all \
  --slice-batch-size 2 \
  --output-kind mean
```

官方 README 的特征抽取要求输入 `[0,1]`，对应 `--normalization minmax`。`--normalization sam` 会额外施加 ImageNet/SAM mean-std，仅作为显式敏感性设置；正式对比必须预注册，不能看过 test 结果后选择。

`all` 最接近 CLARITY 既有 `[D,256]` 约定，但计算成本高。开发探针可用：

```bash
clarity-hauwm extract-mri-core ... \
  --slice-policy uniform --slices-per-modality 16 --limit 2
```

`output-kind tokens` 保存 `[4×slices,256]`；`mean` 直接保存 256 维 timepoint 向量。Stage 1 最小模型最终使用 timepoint 向量，因此两种方式配合 `build-clarity --pooling mean` 数学上等价，后者更省存储。

## 3. 构造两套对齐轨迹

```bash
clarity-hauwm build-clarity \
  --timeline /home/tanyuejun/CLARITY/Predictor/dataset/MU_Glioma_Post/clinical_latest.json \
  --latents data/mri_core_latents \
  --output data/clarity_mri_core \
  --action-anchor source

clarity-hauwm build-clarity \
  --timeline /home/tanyuejun/CLARITY/Predictor/dataset/MU_Glioma_Post/clinical_latest.json \
  --latents data/brainiac_latents \
  --output data/clarity_brainiac \
  --action-anchor source
```

抽取参数和 checkpoint 会嵌入 trajectory provenance，并继续写入训练 checkpoint；用错误 encoder 数据评估时会直接报错。

## 4. Encoder × HS/Ensemble 对比

```bash
clarity-hauwm compare-encoders \
  --encoder-data \
    mri_core=data/clarity_mri_core \
    brainiac=data/clarity_brainiac \
  --config configs/stage1.json \
  --output outputs/encoder_comparison \
  --seeds 7 17 29
```

命令首先验证两套数据具有完全相同的 patient、timepoint、action、`delta_days` 和 action vocabulary；覆盖不一致时拒绝训练。

不同 encoder 的 latent 维度和几何不同，不能按绝对 latent MSE 判定谁更好。`encoder_comparison.json` 只比较各 encoder 内部 `HS+ensemble` 相对其自身 baseline 的误差改善比例、rollout slope 改善和 uncertainty ranking。最终 encoder 选择仍需依赖独立的结局预测或临床 utility 任务。
