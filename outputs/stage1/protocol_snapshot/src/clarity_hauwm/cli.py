from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ablation import evaluate_stage1, train_horizon_ablation, train_split_robustness, train_stage1
from .brainiac_extract import extract_brainiac_latents
from .clarity_adapter import build_clarity_trajectories
from .data import validate_dataset
from .encoder_comparison import parse_encoder_datasets, validate_encoder_alignment
from .mri_core_extract import extract_mri_core_latents
from .reporting import audit_stage1_dataset, summarize_stage1
from .training import TrainingConfig, VARIANTS


def _print(value: dict) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clarity-hauwm", description="CLARITY RRT and ensemble dynamics Stage 1 experiments"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-data", help="Validate trajectory schema")
    validate.add_argument("--data", required=True)

    build = subparsers.add_parser("build-clarity", help="Align CLARITY timeline with frozen MRI latents")
    build.add_argument("--timeline", required=True)
    build.add_argument("--latents", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--action-anchor", choices=("source", "destination"), default="source")
    build.add_argument("--pooling", choices=("mean", "flatten"), default="mean")
    build.add_argument("--min-token-count", type=int, default=1)

    extract = subparsers.add_parser("extract-brainiac", help="Extract frozen BrainIAC latent vectors")
    extract.add_argument("--clarity-root", required=True)
    extract.add_argument("--timeline", required=True)
    extract.add_argument("--mri-root", required=True)
    extract.add_argument("--brainiac-checkpoint", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--device", default="auto")
    extract.add_argument("--tokens-per-modality", type=int, default=8)
    extract.add_argument("--output-kind", choices=("mean", "tokens"), default="mean")

    mri = subparsers.add_parser("extract-mri-core", help="Extract frozen MRI-CORE latent vectors")
    mri.add_argument("--mri-core-root", required=True)
    mri.add_argument("--timeline", required=True)
    mri.add_argument("--mri-root", required=True)
    mri.add_argument("--checkpoint", required=True)
    mri.add_argument("--sam-checkpoint", required=True)
    mri.add_argument("--output", required=True)
    mri.add_argument("--device", default="auto")
    mri.add_argument("--image-size", type=int, default=1024)
    mri.add_argument("--normalization", choices=("minmax", "sam"), default="minmax")
    mri.add_argument("--slice-policy", choices=("all", "uniform"), default="all")
    mri.add_argument("--slices-per-modality", type=int, default=16)
    mri.add_argument("--slice-batch-size", type=int, default=2)
    mri.add_argument("--output-kind", choices=("mean", "tokens"), default="mean")

    align = subparsers.add_parser("validate-encoder-alignment", help="Check matched encoder trajectories")
    align.add_argument("--encoder-data", nargs="+", required=True, metavar="NAME=PATH")

    audit = subparsers.add_parser("audit-stage1", help="Audit the main patient split and horizon windows")
    audit.add_argument("--data", required=True)
    audit.add_argument("--config", required=True)
    audit.add_argument("--output", required=True)

    train = subparsers.add_parser("train-stage1", help="Train independent Stage 1 variants")
    train.add_argument("--data", required=True)
    train.add_argument("--config", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    train.add_argument("--seeds", type=int, nargs="+", default=None)

    robustness = subparsers.add_parser("train-split-robustness", help="Train and evaluate repeated patient splits")
    robustness.add_argument("--data", required=True)
    robustness.add_argument("--config", required=True)
    robustness.add_argument("--output", required=True)

    horizon = subparsers.add_parser("train-horizon-ablation", help="Train K=1/2/3 ablation runs")
    horizon.add_argument("--data", required=True)
    horizon.add_argument("--config", required=True)
    horizon.add_argument("--output", required=True)
    horizon.add_argument("--include-stress", action="store_true",
                         help="Also train K=5 when H4/H5 windows pass the audit threshold")
    horizon.add_argument("--min-stress-windows", type=int, default=30,
                         help="Minimum H4 and H5 windows in each patient split (default: 30)")

    for name in ("evaluate-recursive", "evaluate-uncertainty"):
        evaluate = subparsers.add_parser(name, help=f"Run {name.split('-')[1]} evaluation")
        evaluate.add_argument("--input", required=True)
        evaluate.add_argument("--max-horizon", type=int, default=3)
        evaluate.add_argument("--device", default="auto")

    summary = subparsers.add_parser("summarize-stage1", help="Summarize prediction, reliability, and split robustness")
    summary.add_argument("--input", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "validate-data":
        _print(validate_dataset(args.data))
    elif args.command == "build-clarity":
        _print(build_clarity_trajectories(args.timeline, args.latents, args.output,
                                         action_anchor=args.action_anchor, pooling=args.pooling,
                                         min_token_count=args.min_token_count))
    elif args.command == "extract-brainiac":
        _print(extract_brainiac_latents(
            clarity_root=args.clarity_root, timeline_path=args.timeline, mri_root=args.mri_root,
            brainiac_checkpoint=args.brainiac_checkpoint, output_dir=args.output,
            device_name=args.device, tokens_per_modality=args.tokens_per_modality,
            output_kind=args.output_kind))
    elif args.command == "extract-mri-core":
        _print(extract_mri_core_latents(
            mri_core_root=args.mri_core_root, timeline_path=args.timeline, mri_root=args.mri_root,
            checkpoint_path=args.checkpoint, sam_checkpoint_path=args.sam_checkpoint,
            output_dir=args.output, device_name=args.device, image_size=args.image_size,
            normalization=args.normalization, slice_policy=args.slice_policy,
            slices_per_modality=args.slices_per_modality, slice_batch_size=args.slice_batch_size,
            output_kind=args.output_kind))
    elif args.command == "validate-encoder-alignment":
        _print(validate_encoder_alignment(parse_encoder_datasets(args.encoder_data)))
    elif args.command == "audit-stage1":
        from .evaluation import write_json
        stats = audit_stage1_dataset(args.data, TrainingConfig.from_json(args.config))
        write_json(Path(args.output) / "reports" / "dataset_stats.json", stats)
        _print(stats)
    elif args.command == "train-stage1":
        checkpoints = train_stage1(args.data, args.config, args.output, args.variants, args.seeds)
        print(f"Trained {len(checkpoints)} runs under {Path(args.output).resolve()}")
    elif args.command == "train-split-robustness":
        checkpoints = train_split_robustness(args.data, args.config, args.output)
        print(f"Trained and evaluated {len(checkpoints)} robustness runs under {Path(args.output).resolve()}")
    elif args.command == "train-horizon-ablation":
        checkpoints = train_horizon_ablation(args.data, args.config, args.output,
                                             args.include_stress, args.min_stress_windows)
        print(f"Trained and evaluated {len(checkpoints)} ablation runs under {Path(args.output).resolve()}")
    elif args.command == "evaluate-recursive":
        evaluate_stage1(args.input, "recursive", args.max_horizon, args.device)
    elif args.command == "evaluate-uncertainty":
        evaluate_stage1(args.input, "uncertainty", args.max_horizon, args.device)
    elif args.command == "summarize-stage1":
        summarize_stage1(args.input)
    else:
        raise AssertionError(f"Unhandled command: {args.command}")
