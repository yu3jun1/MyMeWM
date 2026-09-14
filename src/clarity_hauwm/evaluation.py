from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import (
    EvaluationHorizonDataset,
    LatentNormalizer,
    Trajectory,
    collate_windows,
    load_dataset,
    select_trajectories,
)
from .model import EnsembleDynamics, ensemble_mean_and_uncertainty
from .training import load_trained_model, resolve_device


RECORD_FIELDS = ["patient_id", "start", "horizon", "mse", "cosine_distance", "uncertainty"]


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    left = 0
    while left < len(values):
        right = left + 1
        while right < len(values) and sorted_values[right] == sorted_values[left]:
            right += 1
        ranks[order[left:right]] = (left + right - 1) / 2.0
        left = right
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if len(x_array) < 3 or np.std(x_array) == 0 or np.std(y_array) == 0:
        return None
    value = float(np.corrcoef(_average_ranks(x_array), _average_ranks(y_array))[0, 1])
    return value if np.isfinite(value) else None


def _record_metrics(
    patient_ids: Sequence[str],
    starts: torch.Tensor,
    horizons: torch.Tensor,
    mean: torch.Tensor,
    target: torch.Tensor,
    uncertainty: torch.Tensor,
) -> list[dict]:
    mse = (mean - target).square().mean(dim=-1)
    cosine = 1.0 - F.cosine_similarity(mean, target, dim=-1)
    return [
        {
            "patient_id": patient_ids[index],
            "start": int(starts[index]),
            "horizon": int(horizons[index]),
            "mse": float(mse[index]),
            "cosine_distance": float(cosine[index]),
            "uncertainty": float(uncertainty[index]),
        }
        for index in range(len(patient_ids))
    ]


@torch.no_grad()
def evaluate_direct(
    model: EnsembleDynamics,
    trajectories: Sequence[Trajectory],
    normalizer: LatentNormalizer,
    max_horizon: int,
    batch_size: int,
    device: torch.device,
) -> list[dict]:
    dataset = EvaluationHorizonDataset(trajectories, normalizer, max_horizon)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_windows)
    records = []
    for raw_batch in loader:
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in raw_batch.items()
        }
        predictions = model(
            batch["z_start"], batch["actions"], batch["delta_days"], batch["horizon"]
        )
        mean, uncertainty = ensemble_mean_and_uncertainty(predictions)
        records.extend(
            _record_metrics(
                batch["patient_id"],
                batch["start"],
                batch["horizon"],
                mean,
                batch["target"],
                uncertainty,
            )
        )
    return records


@torch.no_grad()
def evaluate_recursive(
    model: EnsembleDynamics,
    trajectories: Sequence[Trajectory],
    normalizer: LatentNormalizer,
    max_horizon: int,
    device: torch.device,
) -> list[dict]:
    records = []
    for trajectory in trajectories:
        initial = torch.from_numpy(normalizer.transform(trajectory.latents[0])).to(device)
        member_states = initial.unsqueeze(0).repeat(model.ensemble_size, 1)
        rollout_length = min(max_horizon, len(trajectory.latents) - 1)
        for step in range(rollout_length):
            actions = torch.from_numpy(trajectory.actions[step : step + 1]).to(device).unsqueeze(0)
            delta_days = torch.from_numpy(trajectory.delta_days[step : step + 1]).to(device).unsqueeze(0)
            member_states = model.forward_memberwise(member_states, actions, delta_days)
            mean, uncertainty = ensemble_mean_and_uncertainty(member_states.unsqueeze(1))
            target = torch.from_numpy(normalizer.transform(trajectory.latents[step + 1])).to(device).unsqueeze(0)
            records.extend(
                _record_metrics(
                    [trajectory.patient_id],
                    torch.tensor([0], device=device),
                    torch.tensor([step + 1], device=device),
                    mean,
                    target,
                    uncertainty,
                )
            )
    return records


def summarize_records(records: Sequence[dict]) -> dict:
    if not records:
        return {"n": 0, "by_horizon": [], "overall": {}}
    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        grouped[int(record["horizon"])].append(record)

    def summarize_group(group: Sequence[dict]) -> dict:
        errors = np.asarray([record["mse"] for record in group])
        uncertainties = np.asarray([record["uncertainty"] for record in group])
        count = len(group)
        quintile_count = max(1, count // 5)
        order = np.argsort(uncertainties)
        low_error = float(errors[order[:quintile_count]].mean())
        high_error = float(errors[order[-quintile_count:]].mean())
        return {
            "n": count,
            "patients": len({record["patient_id"] for record in group}),
            "mse": float(errors.mean()),
            "cosine_distance": float(np.mean([record["cosine_distance"] for record in group])),
            "uncertainty": float(uncertainties.mean()),
            "uncertainty_error_spearman": spearman(uncertainties, errors),
            "high_vs_low_uncertainty_error_ratio": high_error / max(low_error, 1e-12),
        }

    by_horizon = []
    for horizon in sorted(grouped):
        row = summarize_group(grouped[horizon])
        row["horizon"] = horizon
        by_horizon.append(row)
    overall = summarize_group(records)
    if len(by_horizon) >= 2:
        overall["mse_horizon_slope"] = float(
            np.polyfit(
                [row["horizon"] for row in by_horizon],
                [row["mse"] for row in by_horizon],
                deg=1,
            )[0]
        )
        overall["uncertainty_horizon_spearman"] = spearman(
            [record["horizon"] for record in records],
            [record["uncertainty"] for record in records],
        )
    else:
        overall["mse_horizon_slope"] = None
        overall["uncertainty_horizon_spearman"] = None
    long_records = [record for record in records if record["horizon"] >= 2]
    overall["long_horizon_mse"] = (
        float(np.mean([record["mse"] for record in long_records])) if long_records else None
    )
    return {"n": len(records), "by_horizon": by_horizon, "overall": overall}


def write_records(path: str | Path, records: Sequence[dict]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def evaluate_checkpoint(
    data_dir: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    requested_device: str = "auto",
) -> dict:
    device = resolve_device(requested_device)
    model, checkpoint = load_trained_model(checkpoint_path, device)
    trajectories, metadata = load_dataset(data_dir)
    expected = checkpoint["data_metadata"]
    for key in ("schema_version", "latent_dim", "action_dim", "action_vocab"):
        if metadata[key] != expected[key]:
            raise ValueError(f"Dataset/checkpoint mismatch for {key}")
    test_trajectories = select_trajectories(trajectories, checkpoint["split"]["test"])
    normalizer = LatentNormalizer.from_state_dict(checkpoint["normalizer"])
    training_config = checkpoint["training_config"]
    direct_records = evaluate_direct(
        model,
        test_trajectories,
        normalizer,
        model.config.max_horizon,
        training_config["batch_size"],
        device,
    )
    rollout_records = evaluate_recursive(
        model,
        test_trajectories,
        normalizer,
        model.config.max_horizon,
        device,
    )
    report = {
        "variant": checkpoint["variant"],
        "seed": checkpoint["seed"],
        "test_patients": len(test_trajectories),
        "direct": summarize_records(direct_records),
        "rollout": summarize_records(rollout_records),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_records(output_dir / "direct_records.csv", direct_records)
    write_records(output_dir / "rollout_records.csv", rollout_records)
    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report

