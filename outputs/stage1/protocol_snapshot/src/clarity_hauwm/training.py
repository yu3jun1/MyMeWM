from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import (
    EvaluationHorizonDataset, LatentNormalizer, TrainingHorizonDataset,
    collate_windows, load_dataset, select_trajectories, split_patient_ids,
)
from .model import EnsembleDynamics, ModelConfig


VARIANTS = {
    "baseline": {"horizon_strategy": "one_step", "use_ensemble": False},
    "rrt": {"horizon_strategy": "max_available", "use_ensemble": False},
    "ensemble": {"horizon_strategy": "one_step", "use_ensemble": True},
    "rrt_ensemble": {"horizon_strategy": "max_available", "use_ensemble": True},
}


@dataclass(frozen=True)
class TrainingConfig:
    main_max_horizon: int = 3
    horizon_ablation: list[int] = field(default_factory=lambda: [1, 2, 3])
    optional_stress_horizon: int = 5
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    action_embed_dim: int = 32
    time_embed_dim: int = 16
    ensemble_size: int = 5
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 15
    num_workers: int = 0
    main_split_seed: int = 17
    training_seeds: list[int] = field(default_factory=lambda: [7, 17, 29])
    robustness_split_seeds: list[int] = field(default_factory=lambda: [23, 41, 59])
    robustness_training_seed: int = 17
    horizon_ablation_training_seed: int = 17
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    delta_scale_days: float = 365.0
    device: str = "auto"
    terminal_loss_only: bool = True
    teacher_forcing: bool = False
    primary_metric: str = "normalized_latent_mse"
    checkpoint_metric: str = "val_recursive_mse_k2_k3"

    def __post_init__(self) -> None:
        if self.main_max_horizon != 3:
            raise ValueError("Stage 1 main experiment requires main_max_horizon=3")
        if self.horizon_ablation != [1, 2, 3] or self.optional_stress_horizon != 5:
            raise ValueError("Expected horizon_ablation=[1,2,3] and optional_stress_horizon=5")
        if not self.terminal_loss_only or self.teacher_forcing:
            raise ValueError("Stage 1 requires terminal MSE and no teacher forcing")
        if self.primary_metric != "normalized_latent_mse" or self.checkpoint_metric != "val_recursive_mse_k2_k3":
            raise ValueError("Unsupported Stage 1 metric")
        if self.ensemble_size < 1 or self.batch_size < 1 or self.epochs < 1:
            raise ValueError("Invalid positive training parameter")
        if not self.training_seeds or len(set(self.training_seeds)) != len(self.training_seeds):
            raise ValueError("training_seeds must be nonempty and unique")
        if len(set(self.robustness_split_seeds)) != len(self.robustness_split_seeds) or self.main_split_seed in self.robustness_split_seeds:
            raise ValueError("robustness split seeds must be unique and exclude the main split")

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        unknown = set(values) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown training config keys: {sorted(unknown)}")
        return cls(**values)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


@torch.no_grad()
def validation_recursive_mse(model: EnsembleDynamics, loader: DataLoader, device: torch.device) -> tuple[float, dict[int, float]]:
    model.eval()
    totals = {2: 0.0, 3: 0.0}
    counts = {2: 0, 3: 0}
    for raw in loader:
        batch = _move_batch(raw, device)
        prediction = model(batch["z_start"], batch["actions"], batch["delta_days"], batch["horizon"])[0]
        per_window = (prediction - batch["target"]).square().mean(dim=-1)
        for horizon in (2, 3):
            mask = batch["horizon"] == horizon
            totals[horizon] += float(per_window[mask].sum().cpu())
            counts[horizon] += int(mask.sum())
    if not all(counts.values()):
        raise ValueError("Validation split needs windows at both horizons 2 and 3 for checkpoint selection")
    means = {horizon: totals[horizon] / counts[horizon] for horizon in (2, 3)}
    return (means[2] + means[3]) / 2, means


def train_model(data_dir: str | Path, output_dir: str | Path, config: TrainingConfig, seed: int, variant: str,
                split_seed: int | None = None, max_horizon: int | None = None) -> Path:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    switches = VARIANTS[variant]
    training_max_horizon = config.main_max_horizon if max_horizon is None else max_horizon
    if training_max_horizon not in (*config.horizon_ablation, config.optional_stress_horizon):
        raise ValueError(f"Unsupported training max_horizon: {training_max_horizon}")
    device = resolve_device(config.device)
    trajectories, metadata = load_dataset(data_dir)
    provenance = metadata.get("provenance") or {}
    if provenance.get("kind") == "clarity" and provenance.get("action_anchor") != "source":
        raise ValueError("Stage 1 requires CLARITY trajectories with action_anchor=source")
    split_seed = config.main_split_seed if split_seed is None else split_seed
    split = split_patient_ids([item.patient_id for item in trajectories], split_seed,
                              config.train_fraction, config.validation_fraction)
    train_trajectories = select_trajectories(trajectories, split["train"])
    val_trajectories = select_trajectories(trajectories, split["validation"])
    normalizer = LatentNormalizer.fit(train_trajectories)
    validation_dataset = EvaluationHorizonDataset(val_trajectories, normalizer, config.main_max_horizon)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False,
                                   collate_fn=collate_windows)
    members = config.ensemble_size if switches["use_ensemble"] else 1
    model_config = ModelConfig(
        latent_dim=metadata["latent_dim"], action_dim=metadata["action_dim"],
        hidden_dim=config.hidden_dim, action_embed_dim=config.action_embed_dim,
        time_embed_dim=config.time_embed_dim,
        ensemble_size=members, delta_scale_days=config.delta_scale_days,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    member_states = []
    member_reports = []
    for member_index in range(members):
        member_seed = seed if members == 1 else seed * 1000 + member_index
        seed_everything(member_seed)
        train_dataset = TrainingHorizonDataset(train_trajectories, normalizer, training_max_horizon,
                                               switches["horizon_strategy"])
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                  num_workers=config.num_workers, collate_fn=collate_windows,
                                  generator=torch.Generator().manual_seed(member_seed))
        single_config = ModelConfig(**{**model_config.to_dict(), "ensemble_size": 1})
        model = EnsembleDynamics(single_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                      weight_decay=config.weight_decay)
        best_validation = float("inf")
        best_state = None
        best_epoch = 0
        best_train_loss = None
        stale_epochs = 0
        for epoch in range(config.epochs):
            model.train()
            losses = []
            for raw in train_loader:
                batch = _move_batch(raw, device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch["z_start"], batch["actions"], batch["delta_days"], batch["horizon"])[0]
                loss = F.mse_loss(prediction, batch["target"])
                loss.backward()
                if config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            current_validation, _ = validation_recursive_mse(model, validation_loader, device)
            train_loss = float(np.mean(losses))
            print(f"[{variant} seed={seed} member={member_index}] epoch {epoch + 1:03d}/{config.epochs} "
                  f"train={train_loss:.6f} val_recursive_mse_2_3={current_validation:.6f}", flush=True)
            if current_validation < best_validation:
                best_validation = current_validation
                best_epoch = epoch + 1
                best_train_loss = train_loss
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.members[0].state_dict().items()})
                stale_epochs = 0
            else:
                stale_epochs += 1
                if config.early_stopping_patience > 0 and stale_epochs >= config.early_stopping_patience:
                    break
        assert best_state is not None
        member_states.append(best_state)
        member_reports.append({"member": member_index, "member_seed": member_seed, "best_epoch": best_epoch,
                               "train_loss": best_train_loss, "validation_loss": best_validation,
                               "training_horizon_counts": train_dataset.training_horizon_counts()})
    checkpoint_path = output_dir / "best.pt"
    torch.save({
        "schema_version": "4.0", "variant": variant, "seed": seed,
        "model_config": model_config.to_dict(), "training_config": asdict(config),
        "member_state_dicts": member_states, "normalizer": normalizer.state_dict(),
        "split": split,
        "data_metadata": {key: metadata.get(key) for key in
                          ("schema_version", "latent_dim", "action_dim", "action_vocab", "provenance")},
    }, checkpoint_path)
    training = {
        "variant": variant, "seed": seed, "split_seed": split_seed,
        "horizon_strategy": switches["horizon_strategy"],
        "robustness_training_seed": config.robustness_training_seed,
        "data_dir": str(Path(data_dir).resolve()),
        "best_epoch": [member["best_epoch"] for member in member_reports],
        "train_loss": [member["train_loss"] for member in member_reports],
        "validation_loss": [member["validation_loss"] for member in member_reports],
        "max_horizon": training_max_horizon,
        "num_training_windows": len(train_dataset),
        "available_horizon_counts": train_dataset.available_horizon_counts(
            max(config.optional_stress_horizon, training_max_horizon)),
        "training_horizon_counts": train_dataset.training_horizon_counts(),
        "checkpoint": str(checkpoint_path.resolve()), "members": member_reports,
        "checkpoint_metric": config.checkpoint_metric,
        "protocol": {"main_max_horizon": config.main_max_horizon, "ensemble_size": members,
                     "split_seed": split_seed, "train_fraction": config.train_fraction,
                     "validation_fraction": config.validation_fraction},
    }
    with (output_dir / "training.json").open("w", encoding="utf-8") as handle:
        json.dump(training, handle, indent=2)
    return checkpoint_path


def load_trained_model(checkpoint_path: str | Path, device: torch.device) -> tuple[EnsembleDynamics, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != "4.0":
        raise ValueError("Checkpoint is from an older Stage 1 protocol; retrain with train-stage1")
    model = EnsembleDynamics(ModelConfig(**checkpoint["model_config"]))
    if len(checkpoint["member_state_dicts"]) != model.ensemble_size:
        raise ValueError("Checkpoint member count does not match model config")
    for member, state in zip(model.members, checkpoint["member_state_dicts"]):
        member.load_state_dict(state)
    return model.to(device).eval(), checkpoint
