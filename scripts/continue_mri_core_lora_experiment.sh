#!/usr/bin/env bash
set -euo pipefail
cd /home/tanyuejun/CLARITY_HAUWM_Minimal
root=outputs/stage1/mri_core_lora
python_bin=/home/tanyuejun/miniconda3/envs/py310/bin/clarity-hauwm
adapt_pid="$1"
exec 9>"$root/logs/pipeline.lock"
flock -n 9 || exit 1
if [ "$adapt_pid" != "--completed" ]; then
  printf '[%s] Waiting for adaptation process %s\n' "$(date -u +%FT%TZ)" "$adapt_pid"
  while kill -0 "$adapt_pid" 2>/dev/null; do
    sleep 30
  done
fi
if [ ! -s "$root/adaptation/best_lora.pt" ] || [ ! -s "$root/adaptation/adaptation_summary.json" ]; then
  printf '[%s] ERROR: adaptation finished without a complete summary and checkpoint\n' "$(date -u +%FT%TZ)" >&2
  exit 1
fi
printf '[%s] Extracting all adapted MRI-CORE latents\n' "$(date -u +%FT%TZ)"
CUDA_VISIBLE_DEVICES=7 "$python_bin" extract-mri-core-lora --frozen-data data/trajectories/mri_core --output "$root" --resume
printf '[%s] Training six Baseline/RRT dynamics runs\n' "$(date -u +%FT%TZ)"
CUDA_VISIBLE_DEVICES=7 "$python_bin" train-mri-core-lora-dynamics --frozen-data data/trajectories/mri_core --stage1-config configs/stage1_recursive.json --output "$root"
printf '[%s] Summarizing four MRI-CORE groups\n' "$(date -u +%FT%TZ)"
"$python_bin" summarize-mri-core-lora --frozen-data data/trajectories/mri_core --frozen-results outputs/stage1/mri_core --stage1-config configs/stage1_recursive.json --output "$root"
printf '[%s] COMPLETE: %s/reports/summary.md\n' "$(date -u +%FT%TZ)" "$root"
