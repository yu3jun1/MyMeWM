from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import (
    EvaluationHorizonDataset,
    LatentNormalizer,
    TrainingHorizonDataset,
    collate_windows,
    load_dataset,
    select_trajectories,
    split_patient_ids,
)
from .model import EnsembleDynamics, ModelConfig, ensemble_mean_and_uncertainty


@dataclass(frozen=True)
class TrainingConfig:
    max_horizon: int = 5
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    action_embed_dim: int = 32
    time_embed_dim: int = 16
    horizon_embed_dim: int = 16
    ensemble_size: int = 5
    bootstrap_keep_probability: float = 0.8
    cosine_loss_weight: float = 0.1
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 15
    num_workers: int = 0
    split_seed: int = 17
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    delta_scale_days: float = 365.0
    device: str = "auto"

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        known = {field.name for field in fields(cls)}
        unknown = set(values) - known
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


def _prediction_loss(
    predictions: torch.Tensor,
    target: torch.Tensor,
    bootstrap_keep_probability: float,
    cosine_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    member_losses = []
    mse_values = []
    cosine_values = []
    batch_size = target.shape[0]
    for member in range(predictions.shape[0]):
        if predictions.shape[0] == 1 or bootstrap_keep_probability >= 1:
            mask = torch.ones(batch_size, dtype=torch.bool, device=target.device)
        else:
            mask = torch.rand(batch_size, device=target.device) < bootstrap_keep_probability
            if not mask.any():
                mask[member % batch_size] = True
        mse = F.mse_loss(predictions[member, mask], target[mask])
        cosine = (1.0 - F.cosine_similarity(predictions[member, mask], target[mask], dim=-1)).mean()
        member_losses.append(mse + cosine_loss_weight * cosine)
        mse_values.append(mse.detach())
        cosine_values.append(cosine.detach())
    loss = torch.stack(member_losses).mean()
    return loss, {
        "mse": float(torch.stack(mse_values).mean().cpu()),
        "cosine": float(torch.stack(cosine_values).mean().cpu()),
    }


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def validation_mse(model: EnsembleDynamics, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    squared_error_sum = 0.0
    value_count = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        predictions = model(batch["z_start"], batch["actions"], batch["delta_days"], batch["horizon"])
        mean, _ = ensemble_mean_and_uncertainty(predictions)
        squared_error_sum += float((mean - batch["target"]).square().sum().cpu())
        value_count += mean.numel()
    return squared_error_sum / max(value_count, 1)


def train_model(
    data_dir: str | Path,
    output_dir: str | Path,
    config: TrainingConfig,
    seed: int,
    horizon_sampling: bool,
    use_ensemble: bool,
    variant: str,
) -> Path:
    seed_everything(seed)
    device = resolve_device(config.device)
    trajectories, metadata = load_dataset(data_dir)
    split = split_patient_ids(
        [trajectory.patient_id for trajectory in trajectories],
        config.split_seed,
        config.train_fraction,
        config.validation_fraction,
    )
    train_trajectories = select_trajectories(trajectories, split["train"])
    validation_trajectories = select_trajectories(trajectories, split["validation"])
    normalizer = LatentNormalizer.fit(train_trajectories)
    train_dataset = TrainingHorizonDataset(
        train_trajectories,
        normalizer,
        config.max_horizon,
        horizon_sampling,
        seed,
    )
    validation_dataset = EvaluationHorizonDataset(
        validation_trajectories,
        normalizer,
        config.max_horizon,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_windows,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_windows,
    )
    model_config = ModelConfig(
        latent_dim=metadata["latent_dim"],
        action_dim=metadata["action_dim"],
        max_horizon=config.max_horizon,
        hidden_dim=config.hidden_dim,
        action_embed_dim=config.action_embed_dim,
        time_embed_dim=config.time_embed_dim,
        horizon_embed_dim=config.horizon_embed_dim,
        ensemble_size=config.ensemble_size if use_ensemble else 1,
        delta_scale_days=config.delta_scale_days,
    )
    model = EnsembleDynamics(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    best_validation = float("inf")
    stale_epochs = 0
    history = []
    for epoch in range(config.epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        losses = []
        mses = []
        cosines = []
        for raw_batch in train_loader:
            batch = _move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(
                batch["z_start"], batch["actions"], batch["delta_days"], batch["horizon"]
            )
            loss, components = _prediction_loss(
                predictions,
                batch["target"],
                config.bootstrap_keep_probability if use_ensemble else 1.0,
                config.cosine_loss_weight,
            )
            loss.backward()
            if config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            mses.append(components["mse"])
            cosines.append(components["cosine"])
        current_validation = validation_mse(model, validation_loader, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "train_mse": float(np.mean(mses)),
            "train_cosine": float(np.mean(cosines)),
            "validation_mse": current_validation,
        }
        history.append(row)
        print(
            f"[{variant} seed={seed}] epoch {epoch + 1:03d}/{config.epochs} "
            f"train={row['train_loss']:.6f} val_mse={current_validation:.6f}",
            flush=True,
        )
        if current_validation < best_validation:
            best_validation = current_validation
            stale_epochs = 0
            torch.save(
                {
                    "schema_version": "1.0",
                    "variant": variant,
                    "seed": seed,
                    "horizon_sampling": horizon_sampling,
                    "use_ensemble": use_ensemble,
                    "model_config": model_config.to_dict(),
                    "training_config": asdict(config),
                    "model_state_dict": model.state_dict(),
                    "normalizer": normalizer.state_dict(),
                    "split": split,
                    "data_metadata": {
                        "schema_version": metadata["schema_version"],
                        "latent_dim": metadata["latent_dim"],
                        "action_dim": metadata["action_dim"],
                        "action_vocab": metadata["action_vocab"],
                    },
                    "best_validation_mse": best_validation,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if config.early_stopping_patience > 0 and stale_epochs >= config.early_stopping_patience:
                break
    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    return checkpoint_path


def load_trained_model(checkpoint_path: str | Path, device: torch.device) -> tuple[EnsembleDynamics, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = EnsembleDynamics(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint

