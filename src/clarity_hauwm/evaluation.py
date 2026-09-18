from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import EvaluationHorizonDataset, LatentNormalizer, collate_windows, load_dataset, select_trajectories
from .model import EnsembleDynamics, ensemble_mean_and_disagreement
from .training import load_trained_model, resolve_device


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def average_ranks(values: np.ndarray) -> np.ndarray:
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
    value = float(np.corrcoef(average_ranks(x_array), average_ranks(y_array))[0, 1])
    return value if np.isfinite(value) else None


def _load_run(run_dir: Path, requested_device: str, max_horizon: int) -> tuple[EnsembleDynamics, dict, DataLoader, torch.device]:
    with (run_dir / "training.json").open("r", encoding="utf-8") as handle:
        training = json.load(handle)
    device = resolve_device(requested_device)
    model, checkpoint = load_trained_model(training["checkpoint"], device)
    if max_horizon not in (3, 5):
        raise ValueError("Stage 1 evaluation supports H1-H3, or H1-H5 for the optional stress test")
    trajectories, metadata = load_dataset(training["data_dir"])
    for key, expected in checkpoint["data_metadata"].items():
        if metadata.get(key) != expected:
            raise ValueError(f"Dataset/checkpoint mismatch for {key}")
    test = select_trajectories(trajectories, checkpoint["split"]["test"])
    normalizer = LatentNormalizer.from_state_dict(checkpoint["normalizer"])
    dataset = EvaluationHorizonDataset(test, normalizer, max_horizon)
    loader = DataLoader(dataset, batch_size=checkpoint["training_config"]["batch_size"],
                        shuffle=False, collate_fn=collate_windows)
    return model, checkpoint, loader, device


@torch.no_grad()
def evaluate_records(model: EnsembleDynamics, loader: DataLoader, device: torch.device) -> list[dict]:
    records = []
    model.eval()
    for raw in loader:
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                 for key, value in raw.items()}
        predictions = model(batch["z_start"], batch["actions"], batch["delta_days"], batch["horizon"])
        mean, disagreement = ensemble_mean_and_disagreement(predictions)
        mse = (mean - batch["target"]).square().mean(dim=-1)
        cosine = 1.0 - F.cosine_similarity(mean, batch["target"], dim=-1)
        for index, patient_id in enumerate(batch["patient_id"]):
            records.append({
                "patient_id": patient_id, "start": int(batch["start"][index]),
                "horizon": int(batch["horizon"][index]), "mse": float(mse[index]),
                "cosine_distance": float(cosine[index]), "disagreement": float(disagreement[index]),
            })
    return records


def recursive_metrics(records: Sequence[dict], variant: str, seed: int,
                      max_horizon: int = 3) -> dict:
    by_horizon = []
    mse_by_horizon = {}
    for horizon in range(1, max_horizon + 1):
        subset = [row for row in records if row["horizon"] == horizon]
        mse = float(np.mean([row["mse"] for row in subset])) if subset else None
        mse_by_horizon[horizon] = mse
        by_horizon.append({
            "horizon": horizon, "n_predictions": len(subset),
            "n_patients": len({row["patient_id"] for row in subset}),
            "mse": mse,
            "cosine_distance": float(np.mean([row["cosine_distance"] for row in subset])) if subset else None,
        })
    long_mse = ((mse_by_horizon[2] + mse_by_horizon[3]) / 2
                if mse_by_horizon[2] is not None and mse_by_horizon[3] is not None else None)
    return {"variant": variant, "seed": seed, "by_horizon": by_horizon,
            "long_horizon_mse": long_mse}


def _tertile_ratio(uncertainties: Sequence[float], errors: Sequence[float]) -> float | None:
    if len(errors) < 3:
        return None
    count = len(errors) // 3
    order = np.argsort(np.asarray(uncertainties), kind="mergesort")
    values = np.asarray(errors)
    low = float(values[order[:count]].mean())
    high = float(values[order[-count:]].mean())
    return high / low if low > 0 else None


def uncertainty_metrics(records: Sequence[dict], variant: str, seed: int) -> dict:
    by_horizon = []
    for horizon in (1, 2, 3):
        subset = [row for row in records if row["horizon"] == horizon]
        disagreements = [row["disagreement"] for row in subset]
        errors = [row["mse"] for row in subset]
        by_horizon.append({
            "horizon": horizon, "n_predictions": len(subset),
            "mean_disagreement": float(np.mean(disagreements)) if subset else None,
            "disagreement_error_spearman": spearman(disagreements, errors),
            "high_low_tertile_error_ratio": _tertile_ratio(disagreements, errors),
        })
    rhos = [row["disagreement_error_spearman"] for row in by_horizon]
    ratios = [row["high_low_tertile_error_ratio"] for row in by_horizon]
    return {
        "variant": variant, "seed": seed, "by_horizon": by_horizon,
        "macro_spearman": float(np.mean(rhos)) if all(value is not None for value in rhos) else None,
        "macro_high_low_ratio": float(np.mean(ratios)) if all(value is not None for value in ratios) else None,
    }


def selective_risk_metrics(records: Sequence[dict], variant: str, seed: int) -> dict:
    by_horizon = []
    for horizon in (1, 2, 3):
        subset = [row for row in records if row["horizon"] == horizon]
        count = len(subset)
        keep = int(np.floor(count * 0.8))
        risk_100 = float(np.mean([row["mse"] for row in subset])) if count else None
        ranked = sorted(subset, key=lambda row: row["disagreement"])
        risk_80 = float(np.mean([row["mse"] for row in ranked[:keep]])) if keep else None
        reduction = ((risk_100 - risk_80) / risk_100 * 100
                     if risk_100 not in (None, 0) and risk_80 is not None else None)
        by_horizon.append({"horizon": horizon, "n_predictions": count, "n_retained": keep,
                           "risk_100": risk_100, "risk_80": risk_80,
                           "risk_reduction": reduction})
    reductions = [row["risk_reduction"] for row in by_horizon]
    return {"variant": variant, "seed": seed, "by_horizon": by_horizon,
            "macro_risk_reduction": float(np.mean(reductions)) if all(value is not None for value in reductions) else None}


def evaluate_run(run_dir: str | Path, kind: str, max_horizon: int = 3, device: str = "auto") -> dict:
    run_dir = Path(run_dir)
    model, checkpoint, loader, resolved_device = _load_run(run_dir, device, max_horizon)
    records = evaluate_records(model, loader, resolved_device)
    if kind == "recursive":
        report = recursive_metrics(records, checkpoint["variant"], checkpoint["seed"], max_horizon)
        write_json(run_dir / "recursive_records.json", [
            {key: row[key] for key in ("patient_id", "start", "horizon", "mse", "cosine_distance")}
            for row in records
        ])
        write_json(run_dir / "recursive_metrics.json", report)
    elif kind == "uncertainty":
        if max_horizon != 3:
            raise ValueError("Reliability evaluation is restricted to H1-H3")
        if model.ensemble_size < 2:
            raise ValueError("Uncertainty evaluation requires an ensemble variant")
        report = uncertainty_metrics(records, checkpoint["variant"], checkpoint["seed"])
        write_json(run_dir / "uncertainty_records.json", [
            {key: row[key] for key in ("patient_id", "start", "horizon", "mse", "disagreement")}
            for row in records
        ])
        write_json(run_dir / "uncertainty_metrics.json", report)
        write_json(run_dir / "selective_risk.json", selective_risk_metrics(records, checkpoint["variant"], checkpoint["seed"]))
    else:
        raise ValueError(f"Unknown evaluation kind: {kind}")
    return report
