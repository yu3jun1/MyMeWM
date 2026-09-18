from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

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
