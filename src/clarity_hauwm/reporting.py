from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .evaluation import matched_h3_slopes, patient_horizon_means, spearman, write_json
from .training import VARIANTS


def run_directories(root: str | Path, variants: Sequence[str] | None = None) -> list[Path]:
    root = Path(root)
    selected = set(variants or VARIANTS)
    return sorted(path for path in root.glob("seed_*/*")
                  if path.is_dir() and path.name in selected and (path / "training.json").exists())


def _read(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _patient_prediction(root: Path, variant: str) -> tuple[dict[str, dict[int, float]], dict[str, float]]:
    records = []
    slopes: dict[str, list[float]] = {}
    for run in run_directories(root, [variant]):
        current = _read(run / "recursive_records.json")
        records.extend(current)
        for patient, slope in matched_h3_slopes(current).items():
            slopes.setdefault(patient, []).append(slope)
    if not records:
        raise ValueError(f"No recursive records for {variant}; run evaluate-recursive first")
    return patient_horizon_means(records, "mse"), {
        patient: float(np.mean(values)) for patient, values in slopes.items()
    }


def _prediction_value(patients: Sequence[str], values: dict[str, dict[int, float]], metric: str) -> float | None:
    if metric == "matched_h3_slope":
        raise ValueError("Use slope values directly")
    def horizon_mean(horizon: int) -> float | None:
        present = [values[patient][horizon] for patient in patients if patient in values and horizon in values[patient]]
        return float(np.mean(present)) if present else None
    if metric == "long_horizon_mse":
        h2, h3 = horizon_mean(2), horizon_mean(3)
        return (h2 + h3) / 2 if h2 is not None and h3 is not None else None
    return horizon_mean(int(metric[-1]))


def _bootstrap_distribution(point, patients: list[str], samples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(samples):
        sample = [patients[index] for index in rng.integers(0, len(patients), len(patients))]
        baseline, candidate = point(sample)
        if baseline is None or candidate is None:
            continue
        differences.append(baseline - candidate)
    return differences


def _paired_test(baseline: dict[str, dict[int, float]], candidate: dict[str, dict[int, float]],
                 baseline_slopes: dict[str, float], candidate_slopes: dict[str, float],
                 metric: str, samples: int, seed: int) -> dict:
    if metric == "matched_h3_slope":
        patients = sorted(set(baseline_slopes) & set(candidate_slopes))
        def point(ids):
            return float(np.mean([baseline_slopes[patient] for patient in ids])), float(np.mean([candidate_slopes[patient] for patient in ids]))
    else:
        patients = sorted(set(baseline) & set(candidate))
        def point(ids):
            return _prediction_value(ids, baseline, metric), _prediction_value(ids, candidate, metric)
    if len(patients) < 2:
        return {"n_patients": len(patients), "mean_improvement": None, "relative_improvement": None,
                "ci95": [None, None], "probability_of_improvement": None, "valid_bootstrap_samples": 0}
    base, cand = point(patients)
    differences = _bootstrap_distribution(point, patients, samples, seed)
    if base is None or cand is None or len(differences) < samples * 0.95:
        return {"n_patients": len(patients), "mean_improvement": None, "relative_improvement": None,
                "ci95": [None, None], "probability_of_improvement": None,
                "valid_bootstrap_samples": len(differences)}
    return {"n_patients": len(patients), "baseline_mean": base, "candidate_mean": cand,
            "mean_improvement": base - cand,
            "relative_improvement": (base - cand) / abs(base) if base != 0 else None,
            "ci95": [float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))],
            "probability_of_improvement": float(np.mean(np.asarray(differences) > 0)),
            "valid_bootstrap_samples": len(differences)}


def _prediction_summary(values: dict[str, dict[int, float]], slopes: dict[str, float]) -> dict:
    patients = sorted(values)
    horizons = {str(k): _prediction_value(patients, values, f"mse@{k}") for k in (1, 2, 3)}
    return {"mse_by_horizon": horizons,
            "long_horizon_mse": _prediction_value(patients, values, "long_horizon_mse"),
            "matched_h3_slope": float(np.mean(list(slopes.values()))) if slopes else None,
            "matched_h3_patients": len(slopes)}


def _uncertainty_runs(root: Path, variant: str) -> list[dict]:
    runs = []
    for path in run_directories(root, [variant]):
        records = _read(path / "uncertainty_records.json")
        metrics = _read(path / "uncertainty_metrics.json")
        runs.append({"seed": int(metrics["seed"]), "u": patient_horizon_means(records, "uncertainty"),
                     "e": patient_horizon_means(records, "mse"), "metrics": metrics})
    if not runs:
        raise ValueError(f"No uncertainty records for {variant}; run evaluate-uncertainty first")
    return runs


def _rho_values(runs: list[dict], sampled_patients: Sequence[str]) -> dict[str, float | None]:
    per_horizon = {}
    for horizon in (1, 2, 3):
        seed_rhos = []
        for run in runs:
            patients = [patient for patient in sampled_patients if horizon in run["u"].get(patient, {})]
            rho = spearman([run["u"][patient][horizon] for patient in patients],
                           [run["e"][patient][horizon] for patient in patients])
            if rho is None:
                break
            seed_rhos.append(rho)
        per_horizon[str(horizon)] = float(np.mean(seed_rhos)) if len(seed_rhos) == len(runs) else None
    values = list(per_horizon.values())
    per_horizon["macro"] = float(np.mean(values)) if all(value is not None for value in values) else None
    return per_horizon


def _rho_bootstrap(runs: list[dict], samples: int, seed: int) -> dict:
    patients = sorted(set.intersection(*(set(run["u"]) for run in runs)))
    keys = ("1", "2", "3", "macro")
    if not patients:
        return {"n_patients": 0, "rho": {key: None for key in keys},
                "ci95": {key: [None, None] for key in keys},
                "valid_bootstrap_samples": {key: 0 for key in keys}}
    point = _rho_values(runs, patients)
    rng = np.random.default_rng(seed)
    bootstrap = {key: [] for key in keys}
    for _ in range(samples):
        sampled = [patients[index] for index in rng.integers(0, len(patients), len(patients))]
        result = _rho_values(runs, sampled)
        for key in keys:
            if result[key] is not None:
                bootstrap[key].append(result[key])
    cis = {key: ([float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
                 if point[key] is not None and len(values) >= samples * 0.95 else [None, None])
           for key, values in bootstrap.items()}
    return {"n_patients": len(patients), "rho": point, "ci95": cis,
            "valid_bootstrap_samples": {key: len(values) for key, values in bootstrap.items()}}


def _relative(baseline: float | None, candidate: float | None) -> float | None:
    return (baseline - candidate) / abs(baseline) if baseline not in (None, 0) and candidate is not None else None


def summarize_stage1(input_dir: str | Path, bootstrap_samples: int = 2000) -> dict:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    root = Path(input_dir)
    report_dir = root / "reports"
    formal_seeds = {7, 17, 29}
    run_metadata = {variant: [_read(path / "training.json") for path in run_directories(root, [variant])]
                    for variant in VARIANTS}
    data_dirs = {row["data_dir"] for rows in run_metadata.values() for row in rows}
    protocol_complete = (len(data_dirs) == 1 and all(
        {row["seed"] for row in rows} == formal_seeds and len(rows) == 3 and all(
            row.get("protocol") == {"max_horizon": 3,
                                    "ensemble_size": 5 if VARIANTS[variant]["use_ensemble"] else 1,
                                    "split_seed": 17, "train_fraction": 0.7,
                                    "validation_fraction": 0.15}
            for row in rows) for variant, rows in run_metadata.items()))
    patient_predictions = {variant: _patient_prediction(root, variant) for variant in VARIANTS}
    prediction_summary = {variant: _prediction_summary(*patient_predictions[variant]) for variant in VARIANTS}
    paired = {}
    for baseline, candidate in (("baseline", "rhrt"), ("ensemble", "rhrt_ensemble")):
        base, base_slope = patient_predictions[baseline]
        cand, cand_slope = patient_predictions[candidate]
        paired[f"{baseline}_vs_{candidate}"] = {
            metric: _paired_test(base, cand, base_slope, cand_slope, metric, bootstrap_samples, 17 + index)
            for index, metric in enumerate(("mse@2", "mse@3", "long_horizon_mse", "matched_h3_slope"))
        }
    uncertainty_runs = {variant: _uncertainty_runs(root, variant) for variant in ("ensemble", "rhrt_ensemble")}
    uncertainty_summary = {}
    uncertainty_bootstrap = {}
    for variant, runs in uncertainty_runs.items():
        uncertainty_bootstrap[variant] = _rho_bootstrap(runs, bootstrap_samples, 17)
        ratios = {str(horizon): [] for horizon in (1, 2, 3)}
        for run in runs:
            for row in run["metrics"]["by_horizon"]:
                if row["high_low_tertile_error_ratio"] is not None:
                    ratios[str(row["horizon"])].append(row["high_low_tertile_error_ratio"])
        macro_ratios = [run["metrics"]["macro_high_low_ratio"] for run in runs
                        if run["metrics"]["macro_high_low_ratio"] is not None]
        all_ratios_present = len(macro_ratios) == len(runs)
        uncertainty_summary[variant] = {
            "rho_by_horizon": {key: uncertainty_bootstrap[variant]["rho"][key] for key in ("1", "2", "3")},
            "macro_spearman": uncertainty_bootstrap[variant]["rho"]["macro"],
            "high_low_ratio_by_horizon": {key: float(np.mean(values)) if len(values) == len(runs) else None for key, values in ratios.items()},
            "macro_high_low_ratio": float(np.mean(macro_ratios)) if all_ratios_present else None,
            "uncertainty_horizon_spearman": float(np.mean([
                run["metrics"]["uncertainty_horizon_spearman"] for run in runs
                if run["metrics"]["uncertainty_horizon_spearman"] is not None
            ])) if any(run["metrics"]["uncertainty_horizon_spearman"] is not None for run in runs) else None,
        }
    main = paired["baseline_vs_rhrt"]
    long_ci = main["long_horizon_mse"]["ci95"]
    slope_test = main["matched_h3_slope"]
    slope_base = prediction_summary["baseline"]["matched_h3_slope"]
    slope_candidate = prediction_summary["rhrt"]["matched_h3_slope"]
    rhrt_pass = (protocol_complete and long_ci[0] is not None and long_ci[0] > 0 and
                 prediction_summary["rhrt"]["long_horizon_mse"] < prediction_summary["baseline"]["long_horizon_mse"] and
                 slope_test["mean_improvement"] is not None and slope_test["mean_improvement"] > 0 and
                 slope_test["probability_of_improvement"] is not None and slope_test["probability_of_improvement"] > 0.5 and
                 slope_base is not None and slope_candidate is not None and slope_candidate < slope_base)
    ensemble = uncertainty_summary["rhrt_ensemble"]
    rho_ci = uncertainty_bootstrap["rhrt_ensemble"]["ci95"]["macro"]
    ensemble_pass = (protocol_complete and ensemble["macro_spearman"] is not None and ensemble["macro_spearman"] > 0 and
                     rho_ci[0] is not None and rho_ci[0] > 0 and
                     ensemble["macro_high_low_ratio"] is not None and ensemble["macro_high_low_ratio"] > 1)
    summary = {
        "protocol_complete": protocol_complete, "rhrt_pass": rhrt_pass, "ensemble_pass": ensemble_pass,
        "stage1_pass": rhrt_pass and ensemble_pass,
        "long_horizon_relative_improvement": main["long_horizon_mse"]["relative_improvement"],
        "recursive_slope_relative_improvement": _relative(slope_base, slope_candidate),
        "macro_uncertainty_spearman": ensemble["macro_spearman"],
    }
    write_json(report_dir / "rhrt_summary.json", prediction_summary)
    write_json(report_dir / "rhrt_bootstrap.json", paired)
    write_json(report_dir / "ensemble_summary.json", uncertainty_summary)
    write_json(report_dir / "ensemble_bootstrap.json", uncertainty_bootstrap)
    write_json(report_dir / "stage1_summary.json", summary)
    tables = format_tables(prediction_summary, paired, uncertainty_summary)
    (report_dir / "stage1.log").write_text(tables + "\n", encoding="utf-8")
    print(tables)
    return summary


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def format_tables(prediction: dict, paired: dict, uncertainty: dict) -> str:
    lines = ["Recursive Rollout Performance", "| Variant | MSE@1 | MSE@2 | MSE@3 | Long MSE | H3 Slope |",
             "|---|---:|---:|---:|---:|---:|"]
    for variant, row in prediction.items():
        h = row["mse_by_horizon"]
        lines.append(f"| {variant} | {_fmt(h['1'])} | {_fmt(h['2'])} | {_fmt(h['3'])} | "
                     f"{_fmt(row['long_horizon_mse'])} | {_fmt(row['matched_h3_slope'])} |")
    for comparison, tests in paired.items():
        lines.extend(["", f"RHRT Paired Bootstrap: {comparison}",
                      "| Metric | Improvement | Relative | 95% CI | P(Improve) |",
                      "|---|---:|---:|---|---:|"])
        for metric, row in tests.items():
            ci = row["ci95"]
            lines.append(f"| {metric} | {_fmt(row['mean_improvement'])} | {_fmt(row['relative_improvement'])} | "
                         f"[{_fmt(ci[0])}, {_fmt(ci[1])}] | {_fmt(row['probability_of_improvement'])} |")
    for variant, row in uncertainty.items():
        lines.extend(["", f"Ensemble Reliability: {variant}", "| Horizon | Spearman rho | High/Low Tertile Ratio |",
                      "|---:|---:|---:|"])
        for horizon in ("1", "2", "3"):
            lines.append(f"| {horizon} | {_fmt(row['rho_by_horizon'][horizon])} | "
                         f"{_fmt(row['high_low_ratio_by_horizon'][horizon])} |")
        lines.append(f"| Macro | {_fmt(row['macro_spearman'])} | {_fmt(row['macro_high_low_ratio'])} |")
    return "\n".join(lines)
