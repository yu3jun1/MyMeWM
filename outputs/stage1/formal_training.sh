#!/usr/bin/env bash
set -euo pipefail
cd /home/tanyuejun/CLARITY_HAUWM_Minimal
export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
PYTHON=/home/tanyuejun/miniconda3/envs/py310/bin/python
CONFIG=configs/stage1_recursive.json

printf '[%s] Formal Stage 1 training started on physical GPU 7\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
"$PYTHON" -c 'import torch; print(f"Visible CUDA devices: {torch.cuda.device_count()}, device: {torch.cuda.get_device_name(0)}, free/total bytes: {torch.cuda.mem_get_info(0)}", flush=True)'

for seed in 7 17 29; do
  for variant in baseline recursive_max rhrt ensemble rhrt_ensemble; do
    run_dir="outputs/stage1/brainiac/seed_${seed}/${variant}"
    if [[ -s "${run_dir}/training.json" && -s "${run_dir}/best.pt" ]]; then
      printf '[%s] Existing complete BrainIAC run: seed=%s variant=%s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$seed" "$variant"
      continue
    fi
    printf '[%s] Training BrainIAC seed=%s variant=%s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$seed" "$variant"
    "$PYTHON" -m clarity_hauwm train-stage1 --data data/trajectories/brainiac --config "$CONFIG" --output outputs/stage1/brainiac --seeds "$seed" --variants "$variant"
  done
done

printf '[%s] Evaluating BrainIAC main experiment\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
"$PYTHON" -m clarity_hauwm evaluate-recursive --input outputs/stage1/brainiac
"$PYTHON" -m clarity_hauwm evaluate-uncertainty --input outputs/stage1/brainiac
"$PYTHON" -m clarity_hauwm summarize-stage1 --input outputs/stage1/brainiac

printf '[%s] Training BrainIAC split robustness\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
"$PYTHON" -m clarity_hauwm train-split-robustness --data data/trajectories/brainiac --config "$CONFIG" --output outputs/stage1/brainiac
"$PYTHON" -m clarity_hauwm summarize-stage1 --input outputs/stage1/brainiac

for seed in 7 17 29; do
  for variant in baseline rhrt rhrt_ensemble; do
    run_dir="outputs/stage1/mri_core/seed_${seed}/${variant}"
    if [[ -s "${run_dir}/training.json" && -s "${run_dir}/best.pt" ]]; then
      printf '[%s] Existing complete MRI-CORE run: seed=%s variant=%s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$seed" "$variant"
      continue
    fi
    printf '[%s] Training MRI-CORE seed=%s variant=%s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$seed" "$variant"
    "$PYTHON" -m clarity_hauwm train-stage1 --data data/trajectories/mri_core --config "$CONFIG" --output outputs/stage1/mri_core --seeds "$seed" --variants "$variant"
  done
done

printf '[%s] Evaluating MRI-CORE representation robustness\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
"$PYTHON" -m clarity_hauwm evaluate-recursive --input outputs/stage1/mri_core
"$PYTHON" -m clarity_hauwm evaluate-uncertainty --input outputs/stage1/mri_core
"$PYTHON" -m clarity_hauwm summarize-stage1 --input outputs/stage1/mri_core
printf '[%s] Formal Stage 1 workflow complete\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
