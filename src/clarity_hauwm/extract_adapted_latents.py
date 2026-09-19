from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .clarity_adapter import build_clarity_trajectories
from .data import LatentNormalizer, load_dataset, select_trajectories
from .encoder_comparison import validate_encoder_alignment
from .encoders.mri_core_lora import MRICoreLoRAEncoder, load_lora_checkpoint
from .evaluation import write_json
from .lora_adaptation import LoRAAdaptationConfig, _encode_timepoint
from .lora_data import FrozenMRICoreSource, lora_output_root
from .training import resolve_device


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_adapted_latents(
    frozen_data_dir: str | Path,
    output_root: str | Path,
    resume: bool = False,
) -> dict:
    """Freeze the selected LoRA encoder and re-extract every frozen MRI timepoint."""
    root = lora_output_root(output_root)
    checkpoint_path = root / "adaptation" / "best_lora.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"LoRA adaptation checkpoint is missing: {checkpoint_path}")
    checkpoint = load_lora_checkpoint(checkpoint_path)
    config = LoRAAdaptationConfig(**checkpoint["config"])
    stage1 = checkpoint["stage1_config"]
    source = FrozenMRICoreSource(
        frozen_data_dir, config.split_seed,
        stage1["train_fraction"], stage1["validation_fraction"])
    if str(source.data_dir) != checkpoint["source_data_dir"] or source.split != checkpoint["split"]:
        raise ValueError("Frozen dataset or patient split differs from LoRA adaptation")
    if source.extraction != checkpoint["source_extraction"]:
        raise ValueError("Frozen MRI-CORE extraction provenance changed after LoRA adaptation")
    source.assert_preprocessing(
        config.image_size, config.normalization,
        config.slice_policy, config.slices_per_modality)
    files = source.frozen_latent_files()
    features = root / "features"
    latent_dir = features / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)
    progress_path = features / "extraction_progress.json"
    provenance = {
        "base_encoder": "MRI-CORE",
        "adaptation": "LoRA",
        "lora_rank": config.rank,
        "lora_alpha": config.alpha,
        "lora_dropout": config.dropout,
        "encoder_dtype": config.encoder_dtype,
        "distributed_world_size": checkpoint.get("world_size", 1),
        "lora_target": "attention_qv",
        "split_seed": config.split_seed,
        "adaptation_seed": config.adaptation_seed,
        "train_patients_only": True,
        "adapter_checkpoint": str(checkpoint_path.resolve()),
        "adapter_sha256": _file_sha256(checkpoint_path),
        "frozen_data_dir": str(source.data_dir),
        "source_latent_dir": str(source.latent_dir),
        "source_extraction": source.extraction,
    }
    if progress_path.exists():
        previous = json.loads(progress_path.read_text(encoding="utf-8"))
        if not resume or previous["provenance"] != provenance:
            raise ValueError("Existing adapted extraction requires --resume with identical provenance")
    elif any(latent_dir.iterdir()):
        raise FileExistsError(f"Adapted latent directory is nonempty without progress metadata: {latent_dir}")
    else:
        write_json(progress_path, {"provenance": provenance, "completed": 0,
                                   "expected": len(files)})
    device = resolve_device(config.device)
    encoder = MRICoreLoRAEncoder.from_pretrained(
        source.repository, source.checkpoint, source.sam_checkpoint,
        config.image_size, config.rank, config.alpha, config.dropout).to(device)
    encoder.load_adapter_state_dict(checkpoint["adapter_state_dict"])
    encoder.set_compute_dtype(torch.bfloat16 if config.encoder_dtype == "bfloat16" else torch.float32)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    completed = 0
    for frozen_path in files:
        name = frozen_path.stem
        if "_Timepoint_" not in name:
            raise ValueError(f"Unexpected frozen latent filename: {frozen_path}")
        patient_id, number = name.rsplit("_Timepoint_", 1)
        if not number.isdigit():
            raise ValueError(f"Unexpected frozen latent filename: {frozen_path}")
        output_path = latent_dir / frozen_path.name
        if output_path.exists():
            if not resume:
                raise FileExistsError(f"Adapted latent already exists: {output_path}")
            value = np.load(output_path, allow_pickle=False)
            if value.shape != (256,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid resumed adapted latent: {output_path}")
        else:
            latent, _, _ = _encode_timepoint(
                encoder, source, patient_id, f"T{number}", config, device)
            value = latent.detach().cpu().numpy().astype(np.float32)
            if value.shape != (256,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid extracted adapted latent: {output_path}")
            temporary = output_path.with_suffix(".npy.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, value, allow_pickle=False)
            temporary.replace(output_path)
            print(f"LoRA latent {patient_id} T{number}: {tuple(value.shape)}", flush=True)
        completed += 1
        write_json(progress_path, {"provenance": provenance, "completed": completed,
                                   "expected": len(files)})
    extraction_metadata = {
        "encoder": "mri_core_lora",
        "frozen": True,
        "frozen_after_adaptation": True,
        "adapter": "LoRA",
        "encoder_dtype": config.encoder_dtype,
        "train_patients_only": True,
        "adaptation_seed": config.adaptation_seed,
        "split_seed": config.split_seed,
        "adapter_checkpoint": str(checkpoint_path.resolve()),
        "adapter_sha256": provenance["adapter_sha256"],
        "encoder_repository": str(source.repository),
        "encoder_checkpoint": str(source.checkpoint),
        "sam_checkpoint": str(source.sam_checkpoint),
        "timeline": str(source.timeline),
        "mri_root": str(source.mri_root),
        "image_size": config.image_size,
        "normalization": config.normalization,
        "slice_policy": config.slice_policy,
        "slices_per_modality": (
            config.slices_per_modality if config.slice_policy == "uniform" else None),
        "slice_batch_size": config.slice_batch_size,
        "output_kind": "mean",
        "output_dim": 256,
        "extracted": completed,
    }
    write_json(latent_dir / "extraction_metadata.json", extraction_metadata)
    trajectory_dir = features / "trajectories"
    if (trajectory_dir / "metadata.json").exists():
        if not resume:
            raise FileExistsError(f"Adapted trajectories already exist: {trajectory_dir}")
    else:
        build_clarity_trajectories(
            source.timeline, latent_dir, trajectory_dir,
            action_anchor="source",
            pooling="mean",
            min_token_count=source.metadata["provenance"]["min_token_count"])
    alignment = validate_encoder_alignment({
        "frozen": source.data_dir,
        "adapted": trajectory_dir,
    })
    trajectories, _ = load_dataset(trajectory_dir)
    normalizer = LatentNormalizer.fit(select_trajectories(
        trajectories, source.split["train"]))
    write_json(features / "normalization.json", {
        "source": "adapted_latents_of_train_patients_only",
        "split_seed": config.split_seed,
        "train_patient_ids": source.split["train"],
        **normalizer.state_dict(),
    })
    by_patient = {trajectory.patient_id: trajectory for trajectory in trajectories}
    for split_name, patient_ids in source.split.items():
        split_records = []
        for patient_id in patient_ids:
            trajectory = by_patient[patient_id]
            split_records.append({
                "patient_id": patient_id,
                "latents": torch.from_numpy(trajectory.latents.copy()),
                "actions": torch.from_numpy(trajectory.actions.copy()),
                "delta_days": torch.from_numpy(trajectory.delta_days.copy()),
                "timepoints": list(trajectory.timepoints),
            })
        torch.save({"split_seed": config.split_seed, "split": split_name,
                    "patients": split_records}, features / f"{split_name}.pt")
    write_json(features / "provenance.json", provenance)
    summary = {
        "experiment": "mri_core_lora_features",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "adapted_timepoints": completed,
        "adapted_trajectories": len(trajectories),
        "alignment": alignment,
        "normalization_source": "train_patients_only",
        "trajectory_data_dir": str(trajectory_dir.resolve()),
        "latent_dir": str(latent_dir.resolve()),
    }
    write_json(features / "extraction_summary.json", summary)
    return summary
