from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from .encoders.mri_core_lora import MRICoreLoRAEncoder
from .evaluation import write_json
from .lora_data import FrozenMRICoreSource, Transition, lora_output_root
from .model import ModelConfig, OneStepDynamics
from .training import TrainingConfig, resolve_device, seed_everything


@dataclass(frozen=True)
class LoRAAdaptationConfig:
    split_seed: int = 17
    adaptation_seed: int = 17
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    encoder_lr: float = 1e-4
    adaptation_dynamics_lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    epochs: int = 50
    early_stopping_patience: int = 10
    anchor_weight: float = 0.1
    physical_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    slice_batch_size: int = 2
    image_size: int = 1024
    normalization: str = "minmax"
    slice_policy: str = "all"
    slices_per_modality: int = 16
    checkpoint_metric: str = "val_one_step_mse"
    encoder_dtype: str = "bfloat16"
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.rank < 1 or self.alpha <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid LoRA rank, alpha, or dropout")
        if min(self.encoder_lr, self.adaptation_dynamics_lr) <= 0:
            raise ValueError("LoRA learning rates must be positive")
        if self.weight_decay < 0 or self.grad_clip_norm < 0 or self.anchor_weight < 0:
            raise ValueError("Invalid LoRA regularization value")
        if min(self.epochs, self.physical_batch_size,
               self.gradient_accumulation_steps, self.slice_batch_size) < 1:
            raise ValueError("LoRA training sizes and epochs must be positive")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience cannot be negative")
        if self.checkpoint_metric != "val_one_step_mse":
            raise ValueError("LoRA checkpoint selection must use validation one-step MSE")
        if self.encoder_dtype not in ("float32", "bfloat16"):
            raise ValueError("encoder_dtype must be float32 or bfloat16")
        if self.slice_policy not in ("all", "uniform"):
            raise ValueError("Invalid MRI slice policy")

    @classmethod
    def from_json(cls, path: str | Path) -> "LoRAAdaptationConfig":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown MRI-CORE LoRA config keys: {sorted(unknown)}")
        return cls(**values)


@dataclass
class _RNGState:
    cpu: torch.Tensor
    cuda: torch.Tensor | None


def _capture_rng(device: torch.device) -> _RNGState:
    return _RNGState(
        torch.random.get_rng_state().clone(),
        torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None,
    )


def _encode_timepoint(
    encoder: MRICoreLoRAEncoder,
    source: FrozenMRICoreSource,
    patient_id: str,
    timepoint: str,
    config: LoRAAdaptationConfig,
    device: torch.device,
    capture_rng: bool = False,
) -> tuple[torch.Tensor, int, list[_RNGState]]:
    total = None
    count = 0
    states = []
    for inputs in source.iter_preprocessed_slices(
            patient_id, timepoint, config.slice_batch_size,
            config.image_size, config.normalization, config.slice_policy,
            config.slices_per_modality):
        if capture_rng:
            states.append(_capture_rng(device))
        with torch.no_grad():
            tokens = encoder(inputs.to(device=device, dtype=encoder.compute_dtype)).float()
        batch_sum = tokens.sum(dim=0)
        total = batch_sum if total is None else total + batch_sum
        count += len(tokens)
    if count == 0 or total is None:
        raise ValueError(f"No MRI slices for {patient_id} {timepoint}")
    return total / count, count, states


def _replay_source_gradient(
    encoder: MRICoreLoRAEncoder,
    source: FrozenMRICoreSource,
    record: Transition,
    config: LoRAAdaptationConfig,
    device: torch.device,
    states: Sequence[_RNGState],
    count: int,
    gradient: torch.Tensor,
) -> None:
    batches = source.iter_preprocessed_slices(
        record.patient_id, record.source_timepoint, config.slice_batch_size,
        config.image_size, config.normalization, config.slice_policy,
        config.slices_per_modality)
    observed = 0
    device_indexes = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    for inputs, state in zip(batches, states, strict=True):
        with torch.random.fork_rng(devices=device_indexes):
            torch.random.set_rng_state(state.cpu)
            if state.cuda is not None:
                torch.cuda.set_rng_state(state.cuda, device)
            tokens = encoder(inputs.to(device=device, dtype=encoder.compute_dtype)).float()
            tokens.sum(dim=0).backward(gradient / count)
        observed += len(tokens)
    if observed != count:
        raise ValueError("MRI slice count changed between LoRA forward and gradient replay")


def _train_group(
    records: Sequence[Transition],
    encoder: MRICoreLoRAEncoder,
    dynamics: OneStepDynamics,
    source: FrozenMRICoreSource,
    config: LoRAAdaptationConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_denominator: int | None = None,
    synchronize_gradients=None,
) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    denominator = len(records) if loss_denominator is None else loss_denominator
    if not records or denominator < len(records):
        raise ValueError("Invalid distributed LoRA loss denominator")
    losses = []
    dyn_losses = []
    for offset in range(0, len(records), config.physical_batch_size):
        micro = records[offset:offset + config.physical_batch_size]
        source_latents = []
        target_latents = []
        replay = []
        for record in micro:
            encoder.train()
            latent, count, rng_states = _encode_timepoint(
                encoder, source, record.patient_id, record.source_timepoint,
                config, device, capture_rng=True)
            encoder.eval()
            target, _, _ = _encode_timepoint(
                encoder, source, record.patient_id, record.target_timepoint,
                config, device)
            source_latents.append(latent)
            target_latents.append(target)
            replay.append((count, rng_states))
        source_leaf = torch.stack(source_latents).detach().requires_grad_(True)
        target = torch.stack(target_latents).detach()
        frozen = torch.from_numpy(np.stack([row.frozen_source for row in micro])).to(device)
        actions = torch.from_numpy(np.stack([row.action for row in micro])).to(device)
        delta = torch.tensor([row.delta_days for row in micro], dtype=torch.float32, device=device)
        predicted = dynamics(source_leaf, actions, delta)
        per_record_mse = (predicted - target).square().mean(dim=-1)
        per_record_anchor = 1 - F.cosine_similarity(source_leaf, frozen, dim=-1)
        dynamics_loss = per_record_mse.sum() / denominator
        loss = (per_record_mse + config.anchor_weight * per_record_anchor).sum() / denominator
        loss.backward()
        encoder.train()
        for index, record in enumerate(micro):
            count, rng_states = replay[index]
            _replay_source_gradient(
                encoder, source, record, config, device, rng_states,
                count, source_leaf.grad[index].detach())
        losses.append(float(loss.detach().cpu()))
        dyn_losses.append(float(dynamics_loss.detach().cpu()))
    parameters = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    parameters.extend(dynamics.parameters())
    if synchronize_gradients is not None:
        synchronize_gradients(parameters)
    if config.grad_clip_norm:
        torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip_norm)
    optimizer.step()
    return sum(losses), sum(dyn_losses)


@torch.no_grad()
def _validation_mse(
    records: Sequence[Transition],
    encoder: MRICoreLoRAEncoder,
    dynamics: OneStepDynamics,
    source: FrozenMRICoreSource,
    config: LoRAAdaptationConfig,
    device: torch.device,
) -> float:
    encoder.eval()
    dynamics.eval()
    cache: dict[tuple[str, str], torch.Tensor] = {}

    def latent(patient_id: str, timepoint: str) -> torch.Tensor:
        key = patient_id, timepoint
        if key not in cache:
            cache[key] = _encode_timepoint(
                encoder, source, patient_id, timepoint, config, device)[0]
        return cache[key]

    errors = []
    for record in records:
        current = latent(record.patient_id, record.source_timepoint)
        target = latent(record.patient_id, record.target_timepoint)
        action = torch.from_numpy(record.action).to(device).unsqueeze(0)
        delta = torch.tensor([record.delta_days], dtype=torch.float32, device=device)
        predicted = dynamics(current.unsqueeze(0), action, delta)[0]
        errors.append(float((predicted - target).square().mean().cpu()))
    if not errors:
        raise ValueError("Validation split has no one-step transitions")
    return float(np.mean(errors))


def adapt_mri_core_lora(
    frozen_data_dir: str | Path,
    stage1_config_path: str | Path,
    lora_config_path: str | Path,
    output_root: str | Path,
) -> dict:
    output_root = lora_output_root(output_root)
    stage1 = TrainingConfig.from_json(stage1_config_path)
    config = LoRAAdaptationConfig.from_json(lora_config_path)
    if config.split_seed != stage1.main_split_seed:
        raise ValueError("LoRA and Stage 1 patient split seeds differ")
    source = FrozenMRICoreSource(
        frozen_data_dir, config.split_seed,
        stage1.train_fraction, stage1.validation_fraction)
    source.assert_preprocessing(
        config.image_size, config.normalization,
        config.slice_policy, config.slices_per_modality)
    adaptation_dir = output_root / "adaptation"
    distributed = dist.is_available() and dist.is_initialized()
    if not distributed or dist.get_rank() == 0:
        if adaptation_dir.exists() and any(adaptation_dir.iterdir()):
            raise FileExistsError(
                f"LoRA adaptation directory is not empty; use a new mri_core_lora* root: {adaptation_dir}")
        adaptation_dir.mkdir(parents=True, exist_ok=True)
        write_json(adaptation_dir / "config.json", asdict(config))
    if distributed:
        dist.barrier()
    checkpoint_path = adaptation_dir / "best_lora.pt"
    device = resolve_device(config.device)
    seed_everything(config.adaptation_seed)
    encoder = MRICoreLoRAEncoder.from_pretrained(
        source.repository, source.checkpoint, source.sam_checkpoint,
        config.image_size, config.rank, config.alpha, config.dropout).to(device)
    encoder.set_compute_dtype(torch.bfloat16 if config.encoder_dtype == "bfloat16" else torch.float32)
    counts = encoder.parameter_counts()
    print(f"MRI-CORE encoder parameters: {counts}", flush=True)
    dynamics = OneStepDynamics(ModelConfig(
        latent_dim=256, action_dim=source.metadata["action_dim"],
        hidden_dim=stage1.hidden_dim,
        action_embed_dim=stage1.action_embed_dim,
        time_embed_dim=stage1.time_embed_dim,
        delta_scale_days=stage1.delta_scale_days)).to(device)
    optimizer = torch.optim.AdamW([
        {"params": [parameter for parameter in encoder.parameters()
                    if parameter.requires_grad], "lr": config.encoder_lr},
        {"params": dynamics.parameters(), "lr": config.adaptation_dynamics_lr},
    ], weight_decay=config.weight_decay)
    train_records = source.transitions("train")
    val_records = source.transitions("validation")
    if not train_records or not val_records:
        raise ValueError("LoRA adaptation requires train and validation transitions")
    history = []
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    if world_size > config.physical_batch_size * config.gradient_accumulation_steps:
        raise ValueError("Distributed world size exceeds the effective adaptation batch")
    best_score = float("inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, config.epochs + 1):
        shuffled = train_records.copy()
        random.Random(config.adaptation_seed + epoch).shuffle(shuffled)
        encoder.train()
        dynamics.train()
        effective_batch_size = config.physical_batch_size * config.gradient_accumulation_steps
        epoch_total = 0.0
        epoch_dyn_total = 0.0
        for offset in range(0, len(shuffled), effective_batch_size):
            global_group = shuffled[offset:offset + effective_batch_size]
            group = global_group[rank::world_size]
            if not group:
                raise ValueError("Distributed LoRA ranks exceed available train transitions")
            def synchronize(parameters):
                for parameter in parameters:
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
                    accumulated = parameter.grad.detach().cpu()
                    dist.all_reduce(accumulated, op=dist.ReduceOp.SUM)
                    parameter.grad.copy_(accumulated.to(device=parameter.device))

            loss, dyn_loss = _train_group(
                group, encoder, dynamics, source, config,
                device, optimizer,
                loss_denominator=len(global_group),
                synchronize_gradients=synchronize if distributed else None)
            epoch_total += loss * len(global_group)
            epoch_dyn_total += dyn_loss * len(global_group)
        if distributed:
            totals = torch.tensor([epoch_total, epoch_dyn_total], device="cpu")
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            epoch_train_loss = float(totals[0] / len(shuffled))
            epoch_dyn_loss = float(totals[1] / len(shuffled))
        else:
            epoch_train_loss = epoch_total / len(shuffled)
            epoch_dyn_loss = epoch_dyn_total / len(shuffled)
        if distributed:
            dist.barrier()
        validation = _validation_mse(
            val_records, encoder, dynamics, source, config, device) if rank == 0 else 0.0
        if distributed:
            score_tensor = torch.tensor(validation, device="cpu")
            dist.broadcast(score_tensor, src=0)
            validation = float(score_tensor)
        row = {
            "epoch": epoch, "train_adaptation_loss": epoch_train_loss,
            "train_one_step_mse": epoch_dyn_loss,
            "val_one_step_mse": validation,
        }
        history.append(row)
        if rank == 0:
            write_json(adaptation_dir / "training_log.json", history)
            print(f"[MRI-CORE LoRA] epoch {epoch:03d}/{config.epochs} "
                  f"train={row['train_adaptation_loss']:.6f} "
                  f"val_one_step_mse={validation:.6f}", flush=True)
        if validation < best_score:
            best_score = validation
            best_epoch = epoch
            stale_epochs = 0
            checkpoint = {
                "schema_version": "mri_core_lora_1",
                "encoder_dtype": config.encoder_dtype,
                "world_size": world_size,
                "adapter_state_dict": encoder.adapter_state_dict(),
                "dynamics_state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in dynamics.state_dict().items()},
                "config": asdict(config),
                "stage1_config": asdict(stage1),
                "source_data_dir": str(source.data_dir),
                "source_extraction": source.extraction,
                "split": source.split,
                "best_epoch": epoch,
                "best_val_one_step_mse": validation,
                "encoder_parameter_counts": counts,
            }
            if rank == 0:
                temporary = checkpoint_path.with_suffix(".pt.tmp")
                torch.save(checkpoint, temporary)
                temporary.replace(checkpoint_path)
        else:
            stale_epochs += 1
            if config.early_stopping_patience and stale_epochs >= config.early_stopping_patience:
                break
    summary = {
        "experiment": "mri_core_lora_representation_adaptation",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "split_seed": config.split_seed,
        "adaptation_seed": config.adaptation_seed,
        "train_patients_only": True,
        "train_transitions": len(train_records),
        "validation_transitions": len(val_records),
        "best_epoch": best_epoch,
        "best_val_one_step_mse": best_score,
        "checkpoint_metric": config.checkpoint_metric,
        "encoder_parameter_counts": counts,
        "checkpoint": str(checkpoint_path.resolve()),
    }
    summary["encoder_dtype"] = config.encoder_dtype
    summary["distributed_world_size"] = world_size
    if rank == 0:
        write_json(adaptation_dir / "adaptation_summary.json", summary)
        (adaptation_dir / "adaptation_summary.md").write_text(
            "# MRI-CORE LoRA adaptation\n\n"
            f"Train patients only; split seed {config.split_seed}. "
            f"Best epoch {best_epoch}; validation one-step MSE {best_score:.6f}.\n",
            encoding="utf-8")
    return summary
