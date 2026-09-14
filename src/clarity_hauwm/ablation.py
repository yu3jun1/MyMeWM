from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .evaluation import evaluate_checkpoint
from .training import TrainingConfig, train_model


VARIANTS = {
    "baseline": {"horizon_sampling": False, "use_ensemble": False},
    "hs": {"horizon_sampling": True, "use_ensemble": False},
    "ensemble": {"horizon_sampling": False, "use_ensemble": True},
    "hs_ensemble": {"horizon_sampling": True, "use_ensemble": True},
}


def _read_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {
                "patient_id": row["patient_id"],
                "start": int(row["start"]),
                "horizon": int(row["horizon"]),
                "mse": float(row["mse"]),
                "cosine_distance": float(row["cosine_distance"]),
                "uncertainty": float(row["uncertainty"]),
            }
            for row in csv.DictReader(handle)
        ]


def _patient_long_mse(records: Sequence[dict]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record["horizon"] >= 2:
            grouped[record["patient_id"]].append(record["mse"])
    return {patient_id: float(np.mean(values)) for patient_id, values in grouped.items()}


def _patient_slope(records: Sequence[dict]) -> dict[str, float]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["patient_id"]].append(record)
    slopes = {}
    for patient_id, values in grouped.items():
        if len({record["horizon"] for record in values}) >= 2:
            slopes[patient_id] = float(
                np.polyfit(
                    [record["horizon"] for record in values],
                    [record["mse"] for record in values],
                    deg=1,
                )[0]
            )
    return slopes


def _average_patient_metrics(
    paths: Sequence[Path],
    metric: Callable[[Sequence[dict]], dict[str, float]],
) -> dict[str, float]:
    across_runs: dict[str, list[float]] = defaultdict(list)
    for path in paths:
        for patient_id, value in metric(_read_records(path)).items():
            across_runs[patient_id].append(value)
    return {patient_id: float(np.mean(values)) for patient_id, values in across_runs.items()}


def _paired_bootstrap(
    baseline: dict[str, float],
    candidate: dict[str, float],
    seed: int,
    samples: int = 2000,
) -> dict:
    patients = sorted(set(baseline) & set(candidate))
    if len(patients) < 2:
        return {"patients": len(patients), "baseline_mean": None, "candidate_mean": None, "mean_improvement": None, "relative_improvement": None, "ci95": [None, None]}
    baseline_values = np.asarray([baseline[patient] for patient in patients])
    candidate_values = np.asarray([candidate[patient] for patient in patients])
    differences = baseline_values - candidate_values
    rng = np.random.default_rng(seed)
    boot = np.asarray(
        [differences[rng.integers(0, len(differences), size=len(differences))].mean() for _ in range(samples)]
    )
    return {
        "patients": len(patients),
        "baseline_mean": float(baseline_values.mean()),
        "candidate_mean": float(candidate_values.mean()),
        "mean_improvement": float(differences.mean()),
        "relative_improvement": float(differences.mean() / max(abs(baseline_values.mean()), 1e-12)),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "probability_of_improvement": float(np.mean(boot > 0)),
    }


def run_ablation(
    data_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    seeds: Sequence[int],
    bootstrap_samples: int = 2000,
) -> dict:
    if not seeds:
        raise ValueError("At least one seed is required")
    config = TrainingConfig.from_json(config_path)
    output_dir = Path(output_dir)
    reports: dict[str, list[dict]] = {variant: [] for variant in VARIANTS}
    record_paths: dict[str, dict[str, list[Path]]] = {
        variant: {"direct": [], "rollout": []} for variant in VARIANTS
    }
    for seed in seeds:
        for variant, switches in VARIANTS.items():
            run_dir = output_dir / f"seed_{seed}" / variant
            checkpoint = train_model(
                data_dir=data_dir,
                output_dir=run_dir,
                config=config,
                seed=seed,
                variant=variant,
                **switches,
            )
            report = evaluate_checkpoint(
                data_dir=data_dir,
                checkpoint_path=checkpoint,
                output_dir=run_dir,
                requested_device=config.device,
            )
            reports[variant].append(report)
            record_paths[variant]["direct"].append(run_dir / "direct_records.csv")
            record_paths[variant]["rollout"].append(run_dir / "rollout_records.csv")

    baseline_long = _average_patient_metrics(record_paths["baseline"]["direct"], _patient_long_mse)
    candidate_long = _average_patient_metrics(record_paths["hs_ensemble"]["direct"], _patient_long_mse)
    baseline_slope = _average_patient_metrics(record_paths["baseline"]["rollout"], _patient_slope)
    candidate_slope = _average_patient_metrics(record_paths["hs_ensemble"]["rollout"], _patient_slope)
    direct_bootstrap = _paired_bootstrap(
        baseline_long, candidate_long, config.split_seed, bootstrap_samples
    )
    rollout_bootstrap = _paired_bootstrap(
        baseline_slope, candidate_slope, config.split_seed + 1, bootstrap_samples
    )

    uncertainty_correlations = [
        report["direct"]["overall"]["uncertainty_error_spearman"]
        for report in reports["hs_ensemble"]
        if report["direct"]["overall"]["uncertainty_error_spearman"] is not None
    ]
    reliability_ratios = [
        report["direct"]["overall"]["high_vs_low_uncertainty_error_ratio"]
        for report in reports["hs_ensemble"]
    ]
    uncertainty_check = {
        "mean_direct_uncertainty_error_spearman": (
            float(np.mean(uncertainty_correlations)) if uncertainty_correlations else None
        ),
        "mean_direct_high_vs_low_error_ratio": float(np.mean(reliability_ratios)),
    }
    direct_pass = direct_bootstrap["ci95"][0] is not None and direct_bootstrap["ci95"][0] > 0
    rollout_pass = rollout_bootstrap["ci95"][0] is not None and rollout_bootstrap["ci95"][0] > 0
    uncertainty_pass = (
        uncertainty_check["mean_direct_uncertainty_error_spearman"] is not None
        and uncertainty_check["mean_direct_uncertainty_error_spearman"] > 0
        and uncertainty_check["mean_direct_high_vs_low_error_ratio"] > 1
    )
    report = {
        "protocol": {
            "seeds": list(seeds),
            "split_seed": config.split_seed,
            "bootstrap_samples": bootstrap_samples,
            "normalized_latent_metrics": True,
        },
        "runs": reports,
        "confirmatory_tests": {
            "direct_long_horizon_mse": direct_bootstrap,
            "recursive_mse_horizon_slope": rollout_bootstrap,
            "uncertainty_ranking": uncertainty_check,
        },
        "criteria": {
            "direct_long_horizon_pass": direct_pass,
            "recursive_error_growth_pass": rollout_pass,
            "uncertainty_ranking_pass": uncertainty_pass,
            "stage1_pass": direct_pass and rollout_pass and uncertainty_pass,
        },
        "interpretation": (
            "Stage 1 criteria passed; proceed to a separately preregistered outcome/policy study."
            if direct_pass and rollout_pass and uncertainty_pass
            else "Stage 1 criteria did not all pass; do not interpret this world model as a treatment recommender."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "stage1_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report

