#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/tanyuejun/CLARITY_HAUWM_Minimal
SNAPSHOT=$ROOT/outputs/stage1/protocol_snapshot
PYTHON=/home/tanyuejun/miniconda3/envs/py310/bin/python
export PYTHONPATH=$SNAPSHOT/src
export CUDA_VISIBLE_DEVICES=5
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
cd "$ROOT"
finish() {
    result=$?
    echo "$result" > "$ROOT/outputs/stage1/h5_stress.exit"
    echo "K5 stress finished with exit status $result at $(date -u +%FT%TZ)"
}
trap finish EXIT
echo "K5 stress started at $(date -u +%FT%TZ)"
"$PYTHON" - <<'EXPERIMENT'
from pathlib import Path
from clarity_hauwm.training import TrainingConfig, train_model
from clarity_hauwm.evaluation import evaluate_run
from clarity_hauwm.reporting import summarize_stage1

root = Path('/home/tanyuejun/CLARITY_HAUWM_Minimal')
config = TrainingConfig.from_json(root / 'outputs/stage1/protocol_snapshot/configs/stage1_recursive.json')
run = root / 'outputs/stage1/brainiac/horizon_ablation/k5'
train_model(root / 'data/trajectories/brainiac', run, config,
            seed=config.horizon_ablation_training_seed, variant='rrt', max_horizon=5)
report = evaluate_run(run, 'recursive', max_horizon=5, device=config.device)
print('K5 evaluation:', report, flush=True)
summarize_stage1(root / 'outputs/stage1/brainiac')
EXPERIMENT
