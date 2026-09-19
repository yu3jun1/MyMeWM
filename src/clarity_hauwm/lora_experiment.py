from __future__ import annotations

import contextlib
import json
import sys

import torch
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO

from .ablation import evaluate_stage1, train_stage1
from .data import LatentNormalizer, load_dataset, select_trajectories, split_patient_ids
from .encoder_comparison import validate_encoder_alignment
from .lora_data import lora_output_root
from .reporting import summarize_stage1
from .training import TrainingConfig


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def stage_log(output_root: str | Path, stage: str) -> Iterator[Path]:
    root = lora_output_root(output_root)
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{stage}.log"
    with path.open("a", encoding="utf-8") as handle:
        started = datetime.now(timezone.utc).isoformat()
        with contextlib.redirect_stdout(_Tee(sys.stdout, handle)), contextlib.redirect_stderr(
                _Tee(sys.stderr, handle)):
            print(f"[{stage}] started {started}", flush=True)
            try:
                yield path
            finally:
                print(f"[{stage}] ended {datetime.now(timezone.utc).isoformat()}",
                      flush=True)


def train_mri_core_lora_dynamics(
    frozen_data_dir: str | Path,
    stage1_config_path: str | Path,
    output_root: str | Path,
) -> dict:
    root = lora_output_root(output_root)
    adapted_data = root / "features" / "trajectories"
    config = TrainingConfig.from_json(stage1_config_path)
    trajectories, metadata = load_dataset(adapted_data)
    provenance = (metadata.get("provenance") or {}).get("latent_extraction") or {}
    if provenance.get("encoder") != "mri_core_lora" or not provenance.get("train_patients_only"):
        raise ValueError("Dynamics input must be the train-only LoRA-adapted MRI-CORE dataset")
    if provenance.get("split_seed") != config.main_split_seed:
        raise ValueError("Adapted features use a different patient split")
    validate_encoder_alignment({
        "frozen": Path(frozen_data_dir).resolve(), "lora": adapted_data.resolve()})
    split = split_patient_ids(
        [item.patient_id for item in trajectories], config.main_split_seed,
        config.train_fraction, config.validation_fraction)
    normalizer = LatentNormalizer.fit(select_trajectories(
        trajectories, split["train"]))
    feature_provenance = json.loads((root / "features" / "provenance.json").read_text())
    if (feature_provenance["frozen_data_dir"] != str(Path(frozen_data_dir).resolve()) or
            feature_provenance["split_seed"] != config.main_split_seed):
        raise ValueError("Adapted features are not linked to this frozen MRI-CORE dataset")
    from .encoders.mri_core_lora import load_lora_checkpoint
    checkpoint = load_lora_checkpoint(root / "adaptation" / "best_lora.pt")
    original_config = dict(checkpoint["stage1_config"])
    current_config = asdict(config)
    original_config.pop("device", None)
    current_config.pop("device", None)
    if original_config != current_config:
        raise ValueError("Dynamics config differs from the config used during LoRA adaptation")
    # The saved feature statistics must be the train-only statistics used by Stage 1.
    saved = json.loads((root / "features" / "normalization.json").read_text(encoding="utf-8"))
    if (max(abs(a - b) for a, b in zip(saved["mean"], normalizer.mean)) > 1e-5 or
            max(abs(a - b) for a, b in zip(saved["std"], normalizer.std)) > 1e-5):
        raise ValueError("Adapted feature normalization differs from the Stage 1 train split")
    checkpoints = []
    for seed in config.training_seeds:
        for variant in ("baseline", "rrt"):
            run_dir = root / f"seed_{seed}" / variant
            training_path = run_dir / "training.json"
            checkpoint_path = run_dir / "best.pt"
            if training_path.exists() and checkpoint_path.exists():
                training = json.loads(training_path.read_text(encoding="utf-8"))
                if (training["seed"] != seed or training["variant"] != variant or
                        training["split_seed"] != config.main_split_seed or
                        training["max_horizon"] != config.main_max_horizon or
                        training["data_dir"] != str(adapted_data.resolve())):
                    raise ValueError(f"Existing LoRA dynamics run does not match: {run_dir}")
                saved_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                saved_config = dict(saved_checkpoint["training_config"])
                saved_config.pop("device", None)
                if (saved_config != current_config or
                        saved_checkpoint["data_metadata"]["provenance"] != metadata["provenance"] or
                        training["checkpoint"] != str(checkpoint_path.resolve())):
                    raise ValueError(f"Existing LoRA dynamics checkpoint does not match: {run_dir}")
                print(f"Reusing completed LoRA dynamics run: {run_dir}", flush=True)
                checkpoints.append(checkpoint_path)
            elif run_dir.exists() and any(run_dir.iterdir()):
                raise FileExistsError(
                    f"Incomplete LoRA dynamics run requires a new output root: {run_dir}")
            else:
                checkpoints.extend(train_stage1(
                    adapted_data, stage1_config_path, root,
                    variants=(variant,), seeds=(seed,)))
    recursive = evaluate_stage1(root, "recursive", device=config.device)
    summary = summarize_stage1(root)
    return {
        "experiment": "mri_core_lora_dynamics",
        "runs": len(checkpoints),
        "recursive_reports": len(recursive),
        "variants": ["baseline", "rrt"],
        "training_seeds": config.training_seeds,
        "summary": summary,
    }
