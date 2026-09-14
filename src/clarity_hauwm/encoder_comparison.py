from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .ablation import run_ablation
from .data import Trajectory, load_dataset


def parse_encoder_datasets(values: Sequence[str]) -> dict[str, Path]:
    datasets = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or name in datasets:
            raise ValueError(f"Invalid or duplicate encoder name: {name!r}")
        datasets[name] = Path(raw_path).expanduser().resolve()
    if len(datasets) < 2:
        raise ValueError("At least two encoder datasets are required")
    return datasets


def validate_encoder_alignment(dataset_paths: Mapping[str, Path]) -> dict:
    loaded = {name: load_dataset(path) for name, path in dataset_paths.items()}
    reference_name = next(iter(loaded))
    reference_trajectories, reference_metadata = loaded[reference_name]
    reference_by_patient = {trajectory.patient_id: trajectory for trajectory in reference_trajectories}
    for name, (trajectories, metadata) in loaded.items():
        if metadata["action_vocab"] != reference_metadata["action_vocab"]:
            raise ValueError(f"{name}: action vocabulary differs from {reference_name}")
        by_patient = {trajectory.patient_id: trajectory for trajectory in trajectories}
        if set(by_patient) != set(reference_by_patient):
            missing = sorted(set(reference_by_patient) - set(by_patient))[:5]
            extra = sorted(set(by_patient) - set(reference_by_patient))[:5]
            raise ValueError(f"{name}: patient coverage differs; missing={missing}, extra={extra}")
        for patient_id, reference in reference_by_patient.items():
            candidate: Trajectory = by_patient[patient_id]
            if not np.array_equal(candidate.timepoints, reference.timepoints):
                raise ValueError(f"{name}: timepoints differ for {patient_id}")
            if not np.array_equal(candidate.actions, reference.actions):
                raise ValueError(f"{name}: actions differ for {patient_id}")
            if not np.allclose(candidate.delta_days, reference.delta_days):
                raise ValueError(f"{name}: delta_days differ for {patient_id}")
    return {
        "encoders": {
            name: {
                "data_dir": str(dataset_paths[name]),
                "latent_dim": metadata["latent_dim"],
                "provenance": metadata.get("provenance"),
            }
            for name, (_, metadata) in loaded.items()
        },
        "patients": len(reference_trajectories),
        "aligned": True,
    }


def run_encoder_comparison(
    dataset_paths: Mapping[str, Path],
    config_path: str | Path,
    output_dir: str | Path,
    seeds: Sequence[int],
    bootstrap_samples: int = 2000,
) -> dict:
    alignment = validate_encoder_alignment(dataset_paths)
    output_dir = Path(output_dir)
    encoder_reports = {}
    for name, data_path in dataset_paths.items():
        encoder_reports[name] = run_ablation(
            data_dir=data_path,
            config_path=config_path,
            output_dir=output_dir / name,
            seeds=seeds,
            bootstrap_samples=bootstrap_samples,
        )
    within_encoder = {}
    for name, report in encoder_reports.items():
        direct = report["confirmatory_tests"]["direct_long_horizon_mse"]
        rollout = report["confirmatory_tests"]["recursive_mse_horizon_slope"]
        within_encoder[name] = {
            "stage1_pass": report["criteria"]["stage1_pass"],
            "direct_relative_improvement": direct.get("relative_improvement"),
            "recursive_slope_relative_improvement": rollout.get("relative_improvement"),
            **report["confirmatory_tests"]["uncertainty_ranking"],
        }
    comparison = {
        "alignment": alignment,
        "within_encoder_results": within_encoder,
        "reports": encoder_reports,
        "interpretation": (
            "Compare relative HS/ensemble gains and calibration within each encoder. "
            "Do not rank encoders by absolute latent MSE because their latent spaces and dimensions differ."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "encoder_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)
    return comparison
