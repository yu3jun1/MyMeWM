from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ablation import VARIANTS, run_ablation
from .brainiac_extract import extract_brainiac_latents
from .clarity_adapter import build_clarity_trajectories
from .data import validate_dataset
from .encoder_comparison import parse_encoder_datasets, run_encoder_comparison
from .evaluation import evaluate_checkpoint
from .mri_core_extract import extract_mri_core_latents
from .training import TrainingConfig, train_model


def _print(value: dict) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clarity-hauwm",
        description="Minimal CLARITY horizon-sampling and ensemble-dynamics validation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-data", help="Validate trajectory schema and print statistics")
    validate.add_argument("--data", required=True)

    build = subparsers.add_parser("build-clarity", help="Align CLARITY timeline with extracted latent files")
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

    extract_mri_core = subparsers.add_parser(
        "extract-mri-core", help="Extract MRI-CORE 2D slice-token latent vectors"
    )
    extract_mri_core.add_argument("--mri-core-root", required=True)
    extract_mri_core.add_argument("--timeline", required=True)
    extract_mri_core.add_argument("--mri-root", required=True)
    extract_mri_core.add_argument("--checkpoint", required=True)
    extract_mri_core.add_argument("--output", required=True)
    extract_mri_core.add_argument("--device", default="auto")
    extract_mri_core.add_argument("--image-size", type=int, default=1024)
    extract_mri_core.add_argument("--normalization", choices=("minmax", "sam"), default="minmax")
    extract_mri_core.add_argument("--slice-policy", choices=("all", "uniform"), default="all")
    extract_mri_core.add_argument("--slices-per-modality", type=int, default=16)
    extract_mri_core.add_argument("--slice-batch-size", type=int, default=2)
    extract_mri_core.add_argument("--output-kind", choices=("mean", "tokens"), default="mean")

    train = subparsers.add_parser("train", help="Train one ablation variant")
    train.add_argument("--data", required=True)
    train.add_argument("--config", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    train.add_argument("--seed", type=int, default=7)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate direct prediction and recursive rollout")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="auto")

    ablate = subparsers.add_parser("ablate", help="Run all four Stage 1 variants")
    ablate.add_argument("--data", required=True)
    ablate.add_argument("--config", required=True)
    ablate.add_argument("--output", required=True)
    ablate.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29])
    ablate.add_argument("--bootstrap-samples", type=int, default=2000)
    compare = subparsers.add_parser(
        "compare-encoders", help="Run aligned Stage 1 ablations for multiple MRI encoders"
    )
    compare.add_argument(
        "--encoder-data", nargs="+", required=True, metavar="NAME=PATH",
        help="Aligned datasets, e.g. brainiac=data/a mri_core=data/b",
    )
    compare.add_argument("--config", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29])
    compare.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "validate-data":
        _print(validate_dataset(args.data))
    elif args.command == "build-clarity":
        _print(
            build_clarity_trajectories(
                args.timeline,
                args.latents,
                args.output,
                action_anchor=args.action_anchor,
                pooling=args.pooling,
                min_token_count=args.min_token_count,
            )
        )
    elif args.command == "extract-brainiac":
        _print(
            extract_brainiac_latents(
                clarity_root=args.clarity_root,
                timeline_path=args.timeline,
                mri_root=args.mri_root,
                brainiac_checkpoint=args.brainiac_checkpoint,
                output_dir=args.output,
                device_name=args.device,
                tokens_per_modality=args.tokens_per_modality,
                output_kind=args.output_kind,
            )
        )
    elif args.command == "extract-mri-core":
        _print(
            extract_mri_core_latents(
                mri_core_root=args.mri_core_root,
                timeline_path=args.timeline,
                mri_root=args.mri_root,
                checkpoint_path=args.checkpoint,
                output_dir=args.output,
                device_name=args.device,
                image_size=args.image_size,
                normalization=args.normalization,
                slice_policy=args.slice_policy,
                slices_per_modality=args.slices_per_modality,
                slice_batch_size=args.slice_batch_size,
                output_kind=args.output_kind,
            )
        )
    elif args.command == "train":
        config = TrainingConfig.from_json(args.config)
        checkpoint = train_model(
            data_dir=args.data,
            output_dir=args.output,
            config=config,
            seed=args.seed,
            variant=args.variant,
            **VARIANTS[args.variant],
        )
        print(Path(checkpoint).resolve())
    elif args.command == "evaluate":
        _print(evaluate_checkpoint(args.data, args.checkpoint, args.output, args.device))
    elif args.command == "ablate":
        _print(
            run_ablation(
                args.data,
                args.config,
                args.output,
                args.seeds,
                bootstrap_samples=args.bootstrap_samples,
            )
        )
    elif args.command == "compare-encoders":
        _print(
            run_encoder_comparison(
                parse_encoder_datasets(args.encoder_data),
                args.config,
                args.output,
                args.seeds,
                bootstrap_samples=args.bootstrap_samples,
            )
        )
    else:
        raise AssertionError(f"Unhandled command: {args.command}")

