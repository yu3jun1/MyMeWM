from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .data import EvaluationHorizonDataset, LatentNormalizer, load_dataset, select_trajectories, split_patient_ids
from .evaluation import write_json
from .training import TrainingConfig, VARIANTS


METRICS = ("mse@1", "mse@2", "mse@3", "long_mse")
RI_METRICS = {"mse@1": "RI@1", "mse@2": "RI@2", "mse@3": "RI@3", "long_mse": "RI_long"}
COMPARISONS = (("baseline", "recursive_max"), ("baseline", "rhrt"),
               ("recursive_max", "rhrt"), ("ensemble", "rhrt_ensemble"))


def run_directories(root: str | Path, variants: Sequence[str] | None = None) -> list[Path]:
    root = Path(root)
    selected = set(variants or VARIANTS)
    return sorted(path for path in root.glob("seed_*/*")
                  if path.is_dir() and path.name in selected and (path / "training.json").exists())


def _read(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _stats(values: Sequence[float | None]) -> dict[str, float | None]:
    present = [value for value in values if value is not None]
    if len(present) != len(values) or not present:
        return {"mean": None, "std": None}
    return {"mean": float(np.mean(present)),
            "std": float(np.std(present, ddof=1)) if len(present) > 1 else 0.0}


def _prediction_values(report: dict) -> dict[str, float | None]:
    by_horizon = {row["horizon"]: row for row in report["by_horizon"]}
    return {**{f"mse@{h}": by_horizon[h]["mse"] for h in (1, 2, 3)},
            "long_mse": report["long_horizon_mse"],
            **{f"cos@{h}": by_horizon[h]["cosine_distance"] for h in (1, 2, 3)}}


def _relative_percent(base: float | None, candidate: float | None) -> float | None:
    return (base - candidate) / base * 100 if base not in (None, 0) and candidate is not None else None


def _main_runs(root: Path) -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = {}
    for path in run_directories(root):
        training = _read(path / "training.json")
        seed = int(training["seed"])
        variant = training["variant"]
        if variant != path.name or training["horizon_strategy"] != VARIANTS[variant]["horizon_strategy"]:
            raise ValueError(f"Run metadata does not match variant directory: {path}")
        if seed in runs.setdefault(variant, {}):
            raise ValueError(f"Duplicate run for {variant} seed {seed}")
        if not (path / "recursive_metrics.json").exists():
            raise ValueError(f"Missing recursive evaluation for {path}")
        runs[variant][seed] = {"training": training,
                               "prediction": _prediction_values(_read(path / "recursive_metrics.json")),
                               "path": path}
    if not runs:
        raise ValueError(f"No trained runs under {root}")
    data_dirs = {run["training"]["data_dir"] for seeds in runs.values() for run in seeds.values()}
    split_seeds = {run["training"]["split_seed"] for seeds in runs.values() for run in seeds.values()}
    if len(data_dirs) != 1 or len(split_seeds) != 1:
        raise ValueError("Main runs must share a dataset and patient split")
    return runs


def _dataset_stats(training: dict) -> dict:
    trajectories, metadata = load_dataset(training["data_dir"])
    protocol = training["protocol"]
    provenance = metadata.get("provenance") or {}
    if provenance.get("kind") == "clarity" and provenance.get("action_anchor") != "source":
        raise ValueError("Stage 1 requires CLARITY trajectories with action_anchor=source")
    split = split_patient_ids((item.patient_id for item in trajectories), training["split_seed"],
                              protocol["train_fraction"], protocol["validation_fraction"])
    result: dict = {"split_seed": training["split_seed"], "latent_dim": metadata["latent_dim"],
                    "action_dim": metadata["action_dim"], "action_anchor": (metadata.get("provenance") or {}).get("action_anchor"),
                    "window_counts": {}}
    normalizer = LatentNormalizer.fit(select_trajectories(trajectories, split["train"]))
    for name, ids in split.items():
        selected = select_trajectories(trajectories, ids)
        windows = EvaluationHorizonDataset(selected, normalizer, 3).windows
        counts = {str(h): sum(window[2] == h for window in windows) for h in (1, 2, 3)}
        result[f"{name}_patient_count"] = len(selected)
        result[f"{name}_timepoint_count"] = sum(len(item.latents) for item in selected)
        result["window_counts"][name] = counts
    for h in (1, 2, 3):
        result[f"H{h}_window_count"] = result["window_counts"]["test"][str(h)]
    return result


def _prediction_metrics(runs: dict[str, dict[int, dict]]) -> dict:
    result = {}
    for variant in VARIANTS:
        if variant not in runs:
            continue
        seeds = runs[variant]
        result[variant] = {"seeds": {str(seed): run["prediction"] for seed, run in sorted(seeds.items())}}
        for metric in (*METRICS, "cos@1", "cos@2", "cos@3"):
            result[variant][metric] = _stats([run["prediction"][metric] for run in seeds.values()])
    return result


def _prediction_comparison(runs: dict[str, dict[int, dict]]) -> dict:
    result = {}
    for base, candidate in COMPARISONS:
        if base not in runs or candidate not in runs:
            continue
        seeds = sorted(set(runs[base]) & set(runs[candidate]))
        if not seeds:
            continue
        name = f"{base}_vs_{candidate}"
        by_seed = {str(seed): {label: _relative_percent(runs[base][seed]["prediction"][metric],
                                                      runs[candidate][seed]["prediction"][metric])
                               for metric, label in RI_METRICS.items()} for seed in seeds}
        result[name] = {"unit": "percent", "seeds": by_seed}
        for label in RI_METRICS.values():
            result[name][label] = _stats([by_seed[str(seed)][label] for seed in seeds])
    return result


def _training_summary(runs: dict[str, dict[int, dict]]) -> dict:
    result = {}
    for variant, seeds in runs.items():
        result[variant] = {}
        for seed, run in sorted(seeds.items()):
            training = run["training"]
            result[variant][str(seed)] = {
                "training_seed": seed,
                "horizon_strategy": training["horizon_strategy"],
                "best_epoch": training["best_epoch"],
                "best_validation_score": training["validation_loss"],
                "H1_sampling_count": training["horizon_sampling_counts"]["1"],
                "H2_sampling_count": training["horizon_sampling_counts"]["2"],
                "H3_sampling_count": training["horizon_sampling_counts"]["3"],
                "horizon_sampling_percentage": {str(h): (training["horizon_sampling_counts"][str(h)] /
                    sum(training["horizon_sampling_counts"].values()) * 100) for h in (1, 2, 3)},
            }
    return result


def _reliability(runs: dict[str, dict[int, dict]]) -> tuple[dict, dict]:
    uncertainty, selective = {}, {}
    for variant in ("ensemble", "rhrt_ensemble"):
        if variant not in runs:
            continue
        uncertainty_runs, risk_runs = {}, {}
        for seed, run in sorted(runs[variant].items()):
            path = run["path"]
            if not (path / "uncertainty_metrics.json").exists() or not (path / "selective_risk.json").exists():
                raise ValueError(f"Missing uncertainty evaluation for {path}")
            uncertainty_runs[str(seed)] = _read(path / "uncertainty_metrics.json")
            risk_runs[str(seed)] = _read(path / "selective_risk.json")
        uncertainty[variant] = {"seeds": uncertainty_runs}
        selective[variant] = {"seeds": risk_runs}
        for h in (1, 2, 3):
            uncertainty[variant][f"rho@{h}"] = _stats([
                row["by_horizon"][h - 1]["uncertainty_error_spearman"] for row in uncertainty_runs.values()])
            uncertainty[variant][f"high_low_ratio@{h}"] = _stats([
                row["by_horizon"][h - 1]["high_low_tertile_error_ratio"] for row in uncertainty_runs.values()])
            selective[variant][str(h)] = {metric: _stats([
                row["by_horizon"][h - 1][metric] for row in risk_runs.values()])
                for metric in ("risk_100", "risk_80", "risk_reduction")}
        uncertainty[variant]["macro_rho"] = _stats([row["macro_spearman"] for row in uncertainty_runs.values()])
        uncertainty[variant]["macro_high_low_ratio"] = _stats([
            row["macro_high_low_ratio"] for row in uncertainty_runs.values()])
        selective[variant]["macro_risk_reduction"] = _stats([
            row["macro_risk_reduction"] for row in risk_runs.values()])
    return uncertainty, selective


def _robustness(root: Path, runs: dict[str, dict[int, dict]]) -> list[dict]:
    first = next(iter(next(iter(runs.values())).values()))["training"]
    main_split = first["split_seed"]
    robustness_seed = first.get("robustness_training_seed", 17)
    rows = []
    for split_seed in [main_split, *sorted(int(path.name.removeprefix("split_"))
                                          for path in (root / "robustness").glob("split_*"))]:
        values = {}
        for variant in ("baseline", "recursive_max", "rhrt"):
            if split_seed == main_split:
                run = runs.get(variant, {}).get(robustness_seed)
                values[variant] = run["prediction"]["long_mse"] if run else None
            else:
                path = root / "robustness" / f"split_{split_seed}" / f"seed_{robustness_seed}" / variant
                if (path / "recursive_metrics.json").exists():
                    training = _read(path / "training.json")
                    if (training["split_seed"] != split_seed or training["seed"] != robustness_seed or
                            training["data_dir"] != first["data_dir"]):
                        raise ValueError(f"Robustness run metadata mismatch: {path}")
                    values[variant] = _prediction_values(_read(path / "recursive_metrics.json"))["long_mse"]
                else:
                    values[variant] = None
        if any(value is not None for value in values.values()):
            rows.append({"split_seed": split_seed, "training_seed": robustness_seed,
                         "baseline_long_mse": values["baseline"],
                         "recursive_max_long_mse": values["recursive_max"],
                         "rhrt_long_mse": values["rhrt"],
                         "rhrt_relative_improvement": _relative_percent(values["baseline"], values["rhrt"])})
    return rows


def audit_stage1_dataset(data_dir: str | Path, config: TrainingConfig) -> dict:
    stats = _dataset_stats({"data_dir": str(Path(data_dir).resolve()),
                            "split_seed": config.main_split_seed,
                            "protocol": {"train_fraction": config.train_fraction,
                                         "validation_fraction": config.validation_fraction}})
    return stats


def summarize_stage1(input_dir: str | Path) -> dict:
    root = Path(input_dir)
    report_dir = root / "reports"
    runs = _main_runs(root)
    first = next(iter(next(iter(runs.values())).values()))["training"]
    prediction = _prediction_metrics(runs)
    training_summary = _training_summary(runs)
    comparisons = _prediction_comparison(runs)
    uncertainty, selective = _reliability(runs)
    robustness = _robustness(root, runs)
    summary = {
        "main_encoder": root.name,
        "main_split_seed": first["split_seed"],
        "baseline_long_mse": prediction.get("baseline", {}).get("long_mse", {}).get("mean"),
        "recursive_max_long_mse": prediction.get("recursive_max", {}).get("long_mse", {}).get("mean"),
        "rhrt_long_mse": prediction.get("rhrt", {}).get("long_mse", {}).get("mean"),
        "rhrt_relative_improvement": comparisons.get("baseline_vs_rhrt", {}).get("RI_long", {}).get("mean"),
        "rhrt_vs_recursive_max_improvement": comparisons.get("recursive_max_vs_rhrt", {}).get("RI_long", {}).get("mean"),
        "macro_uncertainty_spearman": uncertainty.get("rhrt_ensemble", {}).get("macro_rho", {}).get("mean"),
        "macro_high_low_ratio": uncertainty.get("rhrt_ensemble", {}).get("macro_high_low_ratio", {}).get("mean"),
        "risk_reduction_80": selective.get("rhrt_ensemble", {}).get("macro_risk_reduction", {}).get("mean"),
    }
    reports = {
        "dataset_stats": _dataset_stats(first), "training_summary": training_summary,
        "prediction_metrics": prediction, "prediction_comparison": comparisons,
        "uncertainty_metrics": uncertainty, "selective_risk": selective,
        "split_robustness": robustness, "stage1_summary": summary,
    }
    for name, value in reports.items():
        write_json(report_dir / f"{name}.json", value)
    tables = format_tables(prediction, comparisons, uncertainty, selective, robustness, training_summary)
    (report_dir / "stage1.log").write_text(tables + "\n", encoding="utf-8")
    print(tables)
    return summary


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def _fmt_stats(value: dict) -> str:
    return f"{_fmt(value['mean'])} ± {_fmt(value['std'])}"


def format_tables(prediction: dict, comparisons: dict, uncertainty: dict,
                  selective: dict, robustness: list[dict], training_summary: dict) -> str:
    lines = ["Recursive Rollout Performance (mean ± std across training seeds)",
             "| Variant | MSE@1 | MSE@2 | MSE@3 | Long MSE |",
             "|---|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        if variant in prediction:
            row = prediction[variant]
            lines.append(f"| {variant} | " + " | ".join(_fmt_stats(row[metric]) for metric in METRICS) + " |")
    lines.extend(["", "Relative Improvement (%)", "| Comparison | RI@1 | RI@2 | RI@3 | RI Long |",
                  "|---|---:|---:|---:|---:|"])
    for name, row in comparisons.items():
        lines.append(f"| {name} | " + " | ".join(_fmt_stats(row[label]) for label in RI_METRICS.values()) + " |")
    if "baseline_vs_rhrt" in comparisons:
        lines.extend(["", "Baseline → RHRT by training seed", "| Seed | Baseline Long | RHRT Long | RI Long (%) |",
                      "|---:|---:|---:|---:|"])
        for seed, row in comparisons["baseline_vs_rhrt"]["seeds"].items():
            lines.append(f"| {seed} | {_fmt(prediction['baseline']['seeds'][seed]['long_mse'])} | "
                         f"{_fmt(prediction['rhrt']['seeds'][seed]['long_mse'])} | {_fmt(row['RI_long'])} |")
    for variant, row in uncertainty.items():
        lines.extend(["", f"Ensemble Reliability: {variant}",
                      "| Horizon | Spearman rho | High/Low Ratio | Risk@100 | Risk@80 | Risk Reduction (%) |",
                      "|---:|---:|---:|---:|---:|---:|"])
        for h in (1, 2, 3):
            risk = selective[variant][str(h)]
            lines.append(f"| {h} | {_fmt_stats(row[f'rho@{h}'])} | {_fmt_stats(row[f'high_low_ratio@{h}'])} | "
                         f"{_fmt_stats(risk['risk_100'])} | {_fmt_stats(risk['risk_80'])} | "
                         f"{_fmt_stats(risk['risk_reduction'])} |")
        lines.append(f"| Macro | {_fmt_stats(row['macro_rho'])} | {_fmt_stats(row['macro_high_low_ratio'])} | "
                     f" | | {_fmt_stats(selective[variant]['macro_risk_reduction'])} |")
    if any(variant in training_summary for variant in ("rhrt", "rhrt_ensemble")):
        lines.extend(["", "RHRT Horizon Sampling", "| Variant | Seed | H1 Count (%) | H2 Count (%) | H3 Count (%) |",
                      "|---|---:|---:|---:|---:|"])
        for variant in ("rhrt", "rhrt_ensemble"):
            for seed, row in training_summary.get(variant, {}).items():
                cells = [f"{row[f'H{h}_sampling_count']} ({row['horizon_sampling_percentage'][str(h)]:.1f}%)"
                         for h in (1, 2, 3)]
                lines.append(f"| {variant} | {seed} | " + " | ".join(cells) + " |")
    if robustness:
        lines.extend(["", f"Patient Split Robustness (training seed {robustness[0]['training_seed']})",
                      "| Split Seed | Baseline Long | Recursive-Max Long | RHRT Long | RHRT RI (%) |",
                      "|---:|---:|---:|---:|---:|"])
        for row in robustness:
            lines.append(f"| {row['split_seed']} | {_fmt(row['baseline_long_mse'])} | "
                         f"{_fmt(row['recursive_max_long_mse'])} | {_fmt(row['rhrt_long_mse'])} | "
                         f"{_fmt(row['rhrt_relative_improvement'])} |")
    return "\n".join(lines)
