# 本地 CLARITY 数据就绪度审计（2026-09-14）

只读审计结果：

- `clinical_latest.json`：203 位患者，596 个临床 timepoint；
- 155 位患者至少 2 个 timepoint，110 位至少 3 个，60 位至少 4 个，43 位至少 5 个，最长 6 个；
- MRI 根目录包含 596 个 timepoint 目录，共 2,978 个 `.nii.gz`；理论完整数为 2,980，说明至少有两个模态或 mask 文件缺失，抽取器会跳过不完整 MRI；
- BrainIAC 基础权重存在；
- MRI-CORE 源码存在于 `CLARITY/mri_foundation`，但本地尚未发现 `MRI_CORE_vitb.pth`，需要按官方仓库说明另行放置；
- CLARITY `exp012_cf_diversity` checkpoint 存在，并包含 MRI encoder LoRA 参数；
- `features_output.csv` 当前不存在，所以正式 Stage 1 前必须先运行对应的 `extract-brainiac` 或 `extract-mri-core`。

由此带来的实验约束：

1. `K_max=5` 的最远跨度只来自极少数长轨迹，主结果应同时报告每个 horizon 的患者数，不能只画一条不带样本量的曲线；
2. 推荐把 `k=2..3` 作为 confirmatory long horizon，把 `k=4..5` 标记为 exploratory sensitivity analysis；
3. test split 固定后应检查每个 horizon 的患者覆盖。如果某一 horizon 少于 10 位 test patient，不对该 horizon 单独下显著性结论；
4. checkpoint 的 LoRA 是在同一 CLARITY 任务数据上训练得到的。若严格检验 dynamics 迁移，应同时跑“原始冻结 BrainIAC”和“CLARITY-LoRA BrainIAC”两套 latent，以区分 encoder 任务适配与 HS 的收益。

审计没有读取或输出任何患者级临床内容；这里只记录聚合计数和资产可用性。

