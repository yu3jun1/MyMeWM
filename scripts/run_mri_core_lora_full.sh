#!/usr/bin/env bash
set -euo pipefail
cd /home/tanyuejun/CLARITY_HAUWM_Minimal
root=outputs/stage1/mri_core_lora
mkdir -p "$root/logs"
trap 'printf "[%s] FAILED: inspect %s/logs/launcher.log and stage logs\n" "$(date -u +%FT%TZ)" "$root" >&2' ERR
printf '[%s] Starting three-GPU MRI-CORE LoRA adaptation\n' "$(date -u +%FT%TZ)"
CUDA_VISIBLE_DEVICES=3,6,7 OMP_NUM_THREADS=4 \
  /home/tanyuejun/miniconda3/envs/py310/bin/torchrun \
  --standalone --nnodes=1 --nproc-per-node=3 --no-python \
  /home/tanyuejun/miniconda3/envs/py310/bin/clarity-hauwm \
  adapt-mri-core-lora --frozen-data data/trajectories/mri_core \
  --stage1-config configs/stage1_recursive.json \
  --lora-config configs/mri_core_lora.json --output "$root" \
  > "$root/logs/launcher.log" 2>&1
printf '[%s] Adaptation exited successfully; starting extraction, dynamics, reporting\n' "$(date -u +%FT%TZ)"
bash scripts/continue_mri_core_lora_experiment.sh --completed \
  >> "$root/logs/pipeline.log" 2>&1
printf '[%s] Supplementary experiment complete\n' "$(date -u +%FT%TZ)"
