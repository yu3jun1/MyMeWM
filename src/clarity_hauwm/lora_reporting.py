from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch.nn.functional as F
import torch

from .data import load_dataset, split_patient_ids
from .encoder_comparison import validate_encoder_alignment
from .evaluation import write_json
from .lora_data import lora_output_root
from .reporting import summarize_stage1
from .training import TrainingConfig


METRICS = ("mse@1", "mse@2", "mse@3", "long_mse")


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required experiment result is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}


def _patient_macro(records: list[dict]) -> dict:
    by_horizon = {}
    for horizon in (1, 2, 3):
        patients: dict[str, list[float]] = {}
        for record in records:
            if record["horizon"] == horizon:
                patients.setdefault(record["patient_id"], []).append(record["mse"])
        if not patients:
            by_horizon[str(horizon)] = {"n_patients": 0, "mse": None}
        else:
            by_horizon[str(horizon)] = {
                "n_patients": len(patients),
                "mse": float(np.mean([np.mean(values) for values in patients.values()])),
            }
    second = by_horizon["2"]["mse"]
    third = by_horizon["3"]["mse"]
    return {"by_horizon": by_horizon,
            "long_mse": (second + third) / 2 if second is not None and third is not None else None}


def _all_patient_macro(root: Path, seeds: list[int]) -> dict:
    result = {}
    for variant in ("baseline", "rrt"):
        seed_rows = {}
        for seed in seeds:
            path = root / f"seed_{seed}" / variant / "recursive_records.json"
            seed_rows[str(seed)] = _patient_macro(_read(path))
        result[variant] = {"seeds": seed_rows}
        for horizon in (1, 2, 3):
            values = [row["by_horizon"][str(horizon)]["mse"] for row in seed_rows.values()]
            result[variant][f"mse@{horizon}"] = _stats(values)
        result[variant]["long_mse"] = _stats(
            [row["long_mse"] for row in seed_rows.values()])
    return result


def representation_diagnostics(frozen_data: Path, adapted_data: Path,
                               config: TrainingConfig) -> dict:
    frozen, _ = load_dataset(frozen_data)
    adapted, _ = load_dataset(adapted_data)
    adapted_by_patient = {item.patient_id: item for item in adapted}
    split = split_patient_ids(
        (item.patient_id for item in frozen), config.main_split_seed,
        config.train_fraction, config.validation_fraction)
    result = {}
    for split_name, patient_ids in split.items():
        drift = []
        frozen_smoothness = []
        adapted_smoothness = []
        for patient_id in patient_ids:
            original = next(item for item in frozen if item.patient_id == patient_id)
            updated = adapted_by_patient[patient_id]
            drift.extend(
                1 - F.cosine_similarity(
                    torch.from_numpy(original.latents),
                    torch.from_numpy(updated.latents), dim=-1).numpy())
            frozen_smoothness.extend(
                1 - F.cosine_similarity(
                    torch.from_numpy(original.latents[:-1]),
                    torch.from_numpy(original.latents[1:]), dim=-1).numpy())
            adapted_smoothness.extend(
                1 - F.cosine_similarity(
                    torch.from_numpy(updated.latents[:-1]),
                    torch.from_numpy(updated.latents[1:]), dim=-1).numpy())
        result[split_name] = {
            "n_patients": len(patient_ids),
            "n_timepoints": len(drift),
            "representation_drift_mean": float(np.mean(drift)),
            "frozen_transition_smoothness_mean": float(np.mean(frozen_smoothness)),
            "lora_transition_smoothness_mean": float(np.mean(adapted_smoothness)),
        }
    return result


def summarize_mri_core_lora(
    output_root: str | Path,
    frozen_data_dir: str | Path,
    frozen_results_dir: str | Path,
    stage1_config_path: str | Path,
) -> dict:
    root = lora_output_root(output_root)
    frozen_data = Path(frozen_data_dir).resolve()
    frozen_results = Path(frozen_results_dir).resolve()
    adapted_data = root / "features" / "trajectories"
    if root == frozen_results or adapted_data == frozen_data:
        raise ValueError("LoRA and frozen MRI-CORE directories must be separate")
    config = TrainingConfig.from_json(stage1_config_path)
    alignment = validate_encoder_alignment({
        "frozen": frozen_data, "lora": adapted_data})
    provenance = _read(root / "features" / "provenance.json")
    if (not provenance.get("train_patients_only") or
            provenance.get("split_seed") != config.main_split_seed or
            provenance.get("frozen_data_dir") != str(frozen_data)):
        raise ValueError("LoRA feature provenance does not match the frozen comparison")
    summarize_stage1(root)
    frozen_metrics = _read(frozen_results / "reports" / "prediction_metrics.json")
    adapted_metrics = _read(root / "reports" / "prediction_metrics.json")
    frozen_comparison = _read(frozen_results / "reports" / "prediction_comparison.json")
    adapted_comparison = _read(root / "reports" / "prediction_comparison.json")
    seeds = sorted(config.training_seeds)
    for label, run_root, expected_data in (
        ("frozen", frozen_results, frozen_data),
        ("lora", root, adapted_data.resolve()),
    ):
        for seed in seeds:
            for variant in ("baseline", "rrt"):
                training = _read(run_root / f"seed_{seed}" / variant / "training.json")
                if (training["data_dir"] != str(expected_data) or
                        training["split_seed"] != config.main_split_seed or
                        training["max_horizon"] != config.main_max_horizon):
                    raise ValueError(
                        f"{label} {variant} seed {seed} uses a different dataset or protocol")
    groups = {
        "frozen_baseline": frozen_metrics["baseline"],
        "frozen_rrt": frozen_metrics["rrt"],
        "lora_baseline": adapted_metrics["baseline"],
        "lora_rrt": adapted_metrics["rrt"],
    }
    for name, row in groups.items():
        if sorted(int(seed) for seed in row["seeds"]) != seeds:
            raise ValueError(f"{name} is missing a required training seed")
    four_group = {
        "comparison_scope": "MRI-CORE only; each representation uses train-only latent normalization",
        "training_seeds": seeds,
        "groups": {
            name: {key: row[key] for key in (*METRICS, "seeds")}
            for name, row in groups.items()
        },
    }
    frozen_ri = frozen_comparison["baseline_vs_rrt"]
    lora_ri = adapted_comparison["baseline_vs_rrt"]
    interaction = [
        lora_ri["seeds"][str(seed)]["RI_long"] -
        frozen_ri["seeds"][str(seed)]["RI_long"]
        for seed in seeds
    ]
    paired = {
        "unit": "percent",
        "interaction_unit": "percentage_points",
        "frozen_baseline_vs_rrt": frozen_ri,
        "lora_baseline_vs_rrt": lora_ri,
        "lora_minus_frozen_RI_long_by_seed": {
            str(seed): value for seed, value in zip(seeds, interaction)},
        "lora_minus_frozen_RI_long": _stats(interaction),
    }
    patient_macro = {
        "frozen": _all_patient_macro(frozen_results, seeds),
        "lora": _all_patient_macro(root, seeds),
    }
    diagnostics = representation_diagnostics(frozen_data, adapted_data, config)
    report_dir = root / "reports"
    write_json(report_dir / "four_group_prediction_metrics.json", four_group)
    write_json(report_dir / "paired_relative_improvement.json", paired)
    write_json(report_dir / "patient_macro_metrics.json", patient_macro)
    write_json(report_dir / "representation_diagnostics.json", diagnostics)
    write_json(report_dir / "comparison_provenance.json", {
        "frozen_data_dir": str(frozen_data),
        "frozen_results_dir": str(frozen_results),
        "lora_data_dir": str(adapted_data.resolve()),
        "lora_output_root": str(root),
        "patient_alignment": alignment,
        "feature_provenance": provenance,
    })
    lines = [
        "# MRI-CORE LoRA supplementary experiment",
        "",
        f"Four groups, mean ± sample std across training seeds {seeds}.",
        "MSE values compare dynamics within each MRI-CORE representation; "
        "LoRA changes latent geometry, so absolute cross-representation MSE is descriptive.",
        "",
        "| Representation | Dynamics | MSE@1 | MSE@2 | MSE@3 | Long MSE | RI Long (%) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, representation, variant in (
        ("frozen_baseline", "Frozen", "Baseline"),
        ("frozen_rrt", "Frozen", "RRT"),
        ("lora_baseline", "LoRA", "Baseline"),
        ("lora_rrt", "LoRA", "RRT"),
    ):
        row = groups[name]
        cells = [
            f"{row[metric]['mean']:.4f} ± {row[metric]['std']:.4f}"
            for metric in METRICS
        ]
        ri = (frozen_ri if representation == "Frozen" else lora_ri)["RI_long"]
        ri_text = "—" if variant == "Baseline" else f"{ri['mean']:.2f} ± {ri['std']:.2f}"
        lines.append(f"| {representation} | {variant} | " +
                     " | ".join(cells) + f" | {ri_text} |")
    lines += [
        "",
        "LoRA Baseline vs LoRA RRT is the primary paired comparison.",
        "",
        "## Paired RI Long by seed",
        "",
        "| Seed | Frozen RI (%) | LoRA RI (%) |",
        "|---:|---:|---:|",
    ]
    for seed in seeds:
        lines.append(
            f"| {seed} | {frozen_ri['seeds'][str(seed)]['RI_long']:.2f} | "
            f"{lora_ri['seeds'][str(seed)]['RI_long']:.2f} |")
    lines += [
        "",
        "See patient_macro_metrics.json for patient-weighted errors and "
        "representation_diagnostics.json for drift and transition smoothness.",
        "No automatic pass/fail decision is applied.",
    ]
    text = "\n".join(lines) + "\n"
    (report_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return {
        "frozen_RI_long": frozen_ri["RI_long"],
        "lora_RI_long": lora_ri["RI_long"],
        "lora_minus_frozen_RI_long": paired["lora_minus_frozen_RI_long"],
        "report_dir": str(report_dir.resolve()),
    }
