from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class Trajectory:
    patient_id: str
    latents: np.ndarray
    actions: np.ndarray
    delta_days: np.ndarray
    timepoints: np.ndarray

    def validate(self, latent_dim: int | None = None, action_dim: int | None = None) -> None:
        if self.latents.ndim != 2:
            raise ValueError(f"{self.patient_id}: latents must be [T,D]")
        length, current_latent_dim = self.latents.shape
        if length < 2:
            raise ValueError(f"{self.patient_id}: at least two timepoints are required")
        if self.actions.shape != (length - 1, self.actions.shape[-1]):
            raise ValueError(f"{self.patient_id}: actions must be [T-1,A]")
        if self.delta_days.shape != (length - 1,):
            raise ValueError(f"{self.patient_id}: delta_days must be [T-1]")
        if self.timepoints.shape != (length,):
            raise ValueError(f"{self.patient_id}: timepoints must be [T]")
        if latent_dim is not None and current_latent_dim != latent_dim:
            raise ValueError(f"{self.patient_id}: expected latent_dim={latent_dim}, got {current_latent_dim}")
        if action_dim is not None and self.actions.shape[1] != action_dim:
            raise ValueError(f"{self.patient_id}: expected action_dim={action_dim}, got {self.actions.shape[1]}")
        if not np.isfinite(self.latents).all():
            raise ValueError(f"{self.patient_id}: non-finite latent")
        if not np.isfinite(self.actions).all() or np.any((self.actions < 0) | (self.actions > 1)):
            raise ValueError(f"{self.patient_id}: actions must be finite multi-hot values")
        if not np.isfinite(self.delta_days).all() or np.any(self.delta_days <= 0):
            raise ValueError(f"{self.patient_id}: delta_days must be finite and positive")


@dataclass(frozen=True)
class LatentNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, trajectories: Sequence[Trajectory], min_std: float = 1e-6) -> "LatentNormalizer":
        if not trajectories:
            raise ValueError("Cannot fit normalizer without training trajectories")
        values = np.concatenate([trajectory.latents for trajectory in trajectories], axis=0).astype(np.float64)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std < min_std, 1.0, std)
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform(self, value: np.ndarray) -> np.ndarray:
        return ((value - self.mean) / self.std).astype(np.float32)

    def state_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state_dict(cls, state: dict[str, list[float]]) -> "LatentNormalizer":
        return cls(np.asarray(state["mean"], dtype=np.float32), np.asarray(state["std"], dtype=np.float32))


def _safe_stem(patient_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", patient_id).strip("._")
    if not stem:
        raise ValueError(f"Unsafe empty patient id derived from {patient_id!r}")
    return stem


def save_dataset(
    output_dir: str | Path,
    trajectories: Sequence[Trajectory],
    action_vocab: Sequence[str],
    extra_metadata: dict | None = None,
) -> dict:
    output_dir = Path(output_dir)
    trajectory_dir = output_dir / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    if not trajectories:
        raise ValueError("No trajectories to save")
    latent_dim = trajectories[0].latents.shape[1]
    action_dim = len(action_vocab)
    records = []
    seen_files: set[str] = set()
    for trajectory in trajectories:
        trajectory.validate(latent_dim, action_dim)
        filename = f"{_safe_stem(trajectory.patient_id)}.npz"
        if filename in seen_files:
            suffix = hashlib.sha256(trajectory.patient_id.encode()).hexdigest()[:8]
            filename = f"{_safe_stem(trajectory.patient_id)}_{suffix}.npz"
        seen_files.add(filename)
        relative_path = Path("trajectories") / filename
        np.savez_compressed(
            output_dir / relative_path,
            patient_id=np.asarray(trajectory.patient_id),
            latents=trajectory.latents.astype(np.float32),
            actions=trajectory.actions.astype(np.float32),
            delta_days=trajectory.delta_days.astype(np.float32),
            timepoints=np.asarray(trajectory.timepoints, dtype=str),
        )
        records.append({"patient_id": trajectory.patient_id, "file": relative_path.as_posix()})
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "latent_dim": latent_dim,
        "action_dim": action_dim,
        "action_vocab": list(action_vocab),
        "num_patients": len(trajectories),
        "trajectories": records,
    }
    if extra_metadata:
        metadata["provenance"] = extra_metadata
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return metadata


def load_dataset(data_dir: str | Path) -> tuple[list[Trajectory], dict]:
    data_dir = Path(data_dir)
    with (data_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema version: {metadata.get('schema_version')}")
    trajectories = []
    for record in metadata["trajectories"]:
        path = data_dir / record["file"]
        with np.load(path, allow_pickle=False) as data:
            trajectory = Trajectory(
                patient_id=str(data["patient_id"].item()),
                latents=data["latents"].astype(np.float32),
                actions=data["actions"].astype(np.float32),
                delta_days=data["delta_days"].astype(np.float32),
                timepoints=data["timepoints"].astype(str),
            )
        if trajectory.patient_id != record["patient_id"]:
            raise ValueError(f"Patient id mismatch in {path}")
        trajectory.validate(metadata["latent_dim"], metadata["action_dim"])
        trajectories.append(trajectory)
    if len(trajectories) != metadata["num_patients"]:
        raise ValueError("metadata num_patients does not match loaded trajectories")
    return trajectories, metadata


def split_patient_ids(
    patient_ids: Iterable[str],
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, list[str]]:
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("Invalid split fractions")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be < 1")
    unique_ids = sorted(set(patient_ids))
    if len(unique_ids) < 3:
        raise ValueError("At least three patients are required for train/validation/test")
    ranked = sorted(
        unique_ids,
        key=lambda patient_id: hashlib.sha256(f"{seed}:{patient_id}".encode()).hexdigest(),
    )
    n = len(ranked)
    n_train = max(1, int(n * train_fraction))
    n_validation = max(1, int(n * validation_fraction))
    if n_train + n_validation >= n:
        n_train = n - 2
        n_validation = 1
    return {
        "train": ranked[:n_train],
        "validation": ranked[n_train : n_train + n_validation],
        "test": ranked[n_train + n_validation :],
    }


def select_trajectories(trajectories: Sequence[Trajectory], patient_ids: Sequence[str]) -> list[Trajectory]:
    requested = set(patient_ids)
    selected = [trajectory for trajectory in trajectories if trajectory.patient_id in requested]
    if {trajectory.patient_id for trajectory in selected} != requested:
        raise ValueError("Split contains patient ids that are missing from the dataset")
    return selected


class TrainingHorizonDataset(Dataset):
    """One item per valid start, with the variant's training horizon strategy."""

    def __init__(
        self,
        trajectories: Sequence[Trajectory],
        normalizer: LatentNormalizer,
        max_horizon: int,
        horizon_strategy: str,
        seed: int,
    ) -> None:
        self.trajectories = list(trajectories)
        self.normalizer = normalizer
        self.max_horizon = max_horizon
        if horizon_strategy not in ("one_step", "max_available", "random_available"):
            raise ValueError(f"Unknown horizon strategy: {horizon_strategy}")
        if max_horizon < 1:
            raise ValueError("max_horizon must be positive")
        self.horizon_strategy = horizon_strategy
        self.seed = seed
        self.epoch = 0
        self.starts = [
            (trajectory_index, start)
            for trajectory_index, trajectory in enumerate(self.trajectories)
            for start in range(len(trajectory.latents) - 1)
        ]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.starts)

    def _horizon(self, trajectory_index: int, start: int) -> int:
        if self.horizon_strategy == "one_step":
            return 1
        trajectory = self.trajectories[trajectory_index]
        upper = min(self.max_horizon, len(trajectory.latents) - 1 - start)
        if self.horizon_strategy == "max_available":
            return upper
        digest = hashlib.sha256(f"{self.seed}:{self.epoch}:{trajectory_index}:{start}".encode()).digest()
        return 1 + int.from_bytes(digest[:8], "little") % upper

    def __getitem__(self, index: int) -> dict:
        trajectory_index, start = self.starts[index]
        horizon = self._horizon(trajectory_index, start)
        return make_window(self.trajectories[trajectory_index], start, horizon, self.normalizer)


class EvaluationHorizonDataset(Dataset):
    def __init__(
        self,
        trajectories: Sequence[Trajectory],
        normalizer: LatentNormalizer,
        max_horizon: int,
    ) -> None:
        self.trajectories = list(trajectories)
        self.normalizer = normalizer
        self.windows = [
            (trajectory_index, start, horizon)
            for trajectory_index, trajectory in enumerate(self.trajectories)
            for start in range(len(trajectory.latents) - 1)
            for horizon in range(1, min(max_horizon, len(trajectory.latents) - 1 - start) + 1)
        ]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict:
        trajectory_index, start, horizon = self.windows[index]
        return make_window(self.trajectories[trajectory_index], start, horizon, self.normalizer)


def make_window(
    trajectory: Trajectory,
    start: int,
    horizon: int,
    normalizer: LatentNormalizer,
) -> dict:
    stop = start + horizon
    return {
        "patient_id": trajectory.patient_id,
        "start": start,
        "horizon": horizon,
        "z_start": normalizer.transform(trajectory.latents[start]),
        "target": normalizer.transform(trajectory.latents[stop]),
        "actions": trajectory.actions[start:stop],
        "delta_days": trajectory.delta_days[start:stop],
    }


def collate_windows(items: Sequence[dict]) -> dict:
    batch_size = len(items)
    max_length = max(item["horizon"] for item in items)
    action_dim = items[0]["actions"].shape[1]
    actions = np.zeros((batch_size, max_length, action_dim), dtype=np.float32)
    delta_days = np.zeros((batch_size, max_length), dtype=np.float32)
    for index, item in enumerate(items):
        length = item["horizon"]
        actions[index, :length] = item["actions"]
        delta_days[index, :length] = item["delta_days"]
    return {
        "patient_id": [item["patient_id"] for item in items],
        "start": torch.tensor([item["start"] for item in items], dtype=torch.long),
        "horizon": torch.tensor([item["horizon"] for item in items], dtype=torch.long),
        "z_start": torch.from_numpy(np.stack([item["z_start"] for item in items])),
        "target": torch.from_numpy(np.stack([item["target"] for item in items])),
        "actions": torch.from_numpy(actions),
        "delta_days": torch.from_numpy(delta_days),
    }


def validate_dataset(data_dir: str | Path) -> dict:
    trajectories, metadata = load_dataset(data_dir)
    lengths = np.asarray([len(trajectory.latents) for trajectory in trajectories])
    deltas = np.concatenate([trajectory.delta_days for trajectory in trajectories])
    return {
        "schema_version": metadata["schema_version"],
        "num_patients": len(trajectories),
        "num_timepoints": int(lengths.sum()),
        "num_transitions": int((lengths - 1).sum()),
        "trajectory_length_min": int(lengths.min()),
        "trajectory_length_max": int(lengths.max()),
        "latent_dim": metadata["latent_dim"],
        "action_dim": metadata["action_dim"],
        "delta_days_min": float(deltas.min()),
        "delta_days_max": float(deltas.max()),
    }

