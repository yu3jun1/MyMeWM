from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .evaluation import evaluate_run
from .reporting import run_directories
from .training import TrainingConfig, VARIANTS, train_model


def train_stage1(data_dir: str | Path, config_path: str | Path, output_dir: str | Path,
                 variants: Sequence[str] | None = None, seeds: Sequence[int] | None = None) -> list[Path]:
    config = TrainingConfig.from_json(config_path)
    variants = list(variants or VARIANTS)
    seeds = list(seeds or config.training_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and unique")
    if not variants or len(set(variants)) != len(variants) or set(variants) - set(VARIANTS):
        raise ValueError("Variants must be nonempty, unique, and known")
    checkpoints = []
    for seed in seeds:
        for variant in variants:
            path = Path(output_dir) / f"seed_{seed}" / variant
            checkpoints.append(train_model(data_dir, path, config, seed, variant))
    return checkpoints


def train_split_robustness(data_dir: str | Path, config_path: str | Path,
                           output_dir: str | Path) -> list[Path]:
    config = TrainingConfig.from_json(config_path)
    checkpoints = []
    for split_seed in config.robustness_split_seeds:
        root = Path(output_dir) / "robustness" / f"split_{split_seed}"
        for variant in ("baseline", "recursive_max", "rhrt"):
            path = root / f"seed_{config.robustness_training_seed}" / variant
            checkpoints.append(train_model(data_dir, path, config, config.robustness_training_seed,
                                           variant, split_seed=split_seed))
        evaluate_stage1(root, "recursive", device=config.device)
    return checkpoints


def evaluate_stage1(input_dir: str | Path, kind: str, max_horizon: int = 3,
                    device: str = "auto") -> list[dict]:
    if kind not in ("recursive", "uncertainty"):
        raise ValueError(f"Unknown evaluation kind: {kind}")
    variants = ("ensemble", "rhrt_ensemble") if kind == "uncertainty" else tuple(VARIANTS)
    directories = run_directories(input_dir, variants)
    if not directories:
        raise ValueError(f"No trained runs under {input_dir}")
    reports = [evaluate_run(path, kind, max_horizon, device) for path in directories]
    if kind == "recursive":
        lines = ["Recursive Rollout Performance", "| Seed | Variant | MSE@1 | MSE@2 | MSE@3 | Long MSE |",
                 "|---:|---|---:|---:|---:|---:|"]
        for row in reports:
            h = {item["horizon"]: item["mse"] for item in row["by_horizon"]}
            lines.append(f"| {row['seed']} | {row['variant']} | {_fmt(h[1])} | {_fmt(h[2])} | {_fmt(h[3])} | "
                         f"{_fmt(row['long_horizon_mse'])} |")
    else:
        lines = ["Ensemble Reliability", "| Seed | Variant | rho@1 | rho@2 | rho@3 | Macro rho | Macro High/Low |",
                 "|---:|---|---:|---:|---:|---:|---:|"]
        for row in reports:
            h = {item["horizon"]: item["uncertainty_error_spearman"] for item in row["by_horizon"]}
            lines.append(f"| {row['seed']} | {row['variant']} | {_fmt(h[1])} | {_fmt(h[2])} | {_fmt(h[3])} | "
                         f"{_fmt(row['macro_spearman'])} | {_fmt(row['macro_high_low_ratio'])} |")
    output = "\n".join(lines)
    print(output)
    report_dir = Path(input_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{kind}.log").write_text(output + "\n", encoding="utf-8")
    return reports


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"
