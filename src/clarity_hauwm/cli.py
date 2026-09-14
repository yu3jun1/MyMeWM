from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ablation import VARIANTS, run_ablation
from .brainiac_extract import extract_brainiac_latents
from .clarity_adapter import build_clarity_trajectories
from .data import validate_dataset
from .evaluation import evaluate_checkpoint
from .synthetic import generate_synthetic_dataset
from .training import TrainingConfig, train_model


def _print(value: dict) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clarity-hauwm",
        description="Minimal CLARITY horizon-sampling and ensemble-dynamics validation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthesize = subparsers.add_parser("synthesize", help="Generate non-clinical smoke-test trajectories")
    synthesize.add_argument("--output", required=True)
    synthesize.add_argument("--patients", type=int, default=80)
    synthesize.add_argument("--latent-dim", type=int, default=8)
    synthesize.add_argument("--action-dim", type=int, default=4)
    synthesize.add_argument("--min-timepoints", type=int, default=3)
    synthesize.add_argument("--max-timepoints", type=int, default=6)
    synthesize.add_argument("--seed", type=int, default=7)

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
    extract.add_argument("--clarity-checkpoint")
    extract.add_argument("--output", required=True)
    extract.add_argument("--device", default="auto")
    extract.add_argument("--tokens-per-modality", type=int, default=8)
    extract.add_argument("--output-kind", choices=("mean", "tokens"), default="mean")
    extract.add_argument("--limit", type=int)

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
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "synthesize":
        _print(
            generate_synthetic_dataset(
                args.output,
                patients=args.patients,
                latent_dim=args.latent_dim,
                action_dim=args.action_dim,
                min_timepoints=args.min_timepoints,
                max_timepoints=args.max_timepoints,
                seed=args.seed,
            )
        )
    elif args.command == "validate-data":
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
                clarity_checkpoint=args.clarity_checkpoint,
                output_dir=args.output,
                device_name=args.device,
                tokens_per_modality=args.tokens_per_modality,
                output_kind=args.output_kind,
                limit=args.limit,
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
    else:
        raise AssertionError(f"Unhandled command: {args.command}")

