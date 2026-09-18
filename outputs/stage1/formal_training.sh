#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/tanyuejun/CLARITY_HAUWM_Minimal
SNAPSHOT=$ROOT/outputs/stage1/protocol_snapshot
PYTHON=/home/tanyuejun/miniconda3/envs/py310/bin/python
CONFIG=$SNAPSHOT/configs/stage1_recursive.json
export PYTHONPATH=$SNAPSHOT/src
export CUDA_VISIBLE_DEVICES=5
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
cd "$ROOT"
finish() {
    result=$?
    echo "$result" > "$ROOT/outputs/stage1/formal_training.exit"
    echo "Finished with exit status $result at $(date -u +%FT%TZ)"
}
trap finish EXIT
echo "Formal Stage 1 started at $(date -u +%FT%TZ)"
echo "Using physical GPU 5, source snapshot $SNAPSHOT"
echo "Phase A: Data audit"
"$PYTHON" -m clarity_hauwm audit-stage1 --data data/trajectories/brainiac --config "$CONFIG" --output outputs/stage1/brainiac
"$PYTHON" -m clarity_hauwm audit-stage1 --data data/trajectories/mri_core --config "$CONFIG" --output outputs/stage1/mri_core
echo "Phase B: BrainIAC main experiment"
"$PYTHON" -m clarity_hauwm train-stage1 --data data/trajectories/brainiac --config "$CONFIG" --output outputs/stage1/brainiac
"$PYTHON" -m clarity_hauwm evaluate-recursive --input outputs/stage1/brainiac
"$PYTHON" -m clarity_hauwm evaluate-uncertainty --input outputs/stage1/brainiac
echo "Phase C: BrainIAC patient split robustness"
"$PYTHON" -m clarity_hauwm train-split-robustness --data data/trajectories/brainiac --config "$CONFIG" --output outputs/stage1/brainiac
echo "Phase D: BrainIAC training horizon ablation"
"$PYTHON" -m clarity_hauwm train-horizon-ablation --data data/trajectories/brainiac --config "$CONFIG" --output outputs/stage1/brainiac
"$PYTHON" -m clarity_hauwm summarize-stage1 --input outputs/stage1/brainiac
echo "Phase F: MRI-CORE representation robustness"
"$PYTHON" -m clarity_hauwm train-stage1 --data data/trajectories/mri_core --config "$CONFIG" --variants baseline rrt rrt_ensemble --output outputs/stage1/mri_core
"$PYTHON" -m clarity_hauwm evaluate-recursive --input outputs/stage1/mri_core
"$PYTHON" -m clarity_hauwm evaluate-uncertainty --input outputs/stage1/mri_core
"$PYTHON" -m clarity_hauwm summarize-stage1 --input outputs/stage1/mri_core
echo "All formal Stage 1 phases completed at $(date -u +%FT%TZ)"
