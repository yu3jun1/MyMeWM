from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .clarity_adapter import latent_filename
from .training import resolve_device


def extract_brainiac_latents(
    clarity_root: str | Path,
    timeline_path: str | Path,
    mri_root: str | Path,
    brainiac_checkpoint: str | Path,
    output_dir: str | Path,
    clarity_checkpoint: str | Path | None = None,
    device_name: str = "auto",
    tokens_per_modality: int = 8,
    output_kind: str = "mean",
    limit: int | None = None,
) -> dict:
    clarity_root = Path(clarity_root).resolve()
    predictor_root = clarity_root / "Predictor"
    if str(predictor_root) not in sys.path:
        sys.path.insert(0, str(predictor_root))
    if str(clarity_root) not in sys.path:
        sys.path.insert(0, str(clarity_root))
    brainiac_module = importlib.import_module("models.brainiac_adapter")
    vision_module = importlib.import_module("models.vision_backbone")
    dataset_module = importlib.import_module("dataset.dataset_glioma_all_pairs_text")

    use_lora = clarity_checkpoint is not None
    backbone = brainiac_module.BrainIACAdapter(
        checkpoint_path=str(brainiac_checkpoint),
        tokens_per_modality=tokens_per_modality,
        lora_r=8 if use_lora else 0,
    )
    encoder = vision_module.MultiModalVisionBackbone(backbone, num_modalities=4)
    loaded_lora_tensors = 0
    if clarity_checkpoint is not None:
        checkpoint = torch.load(clarity_checkpoint, map_location="cpu", weights_only=True)
        state = checkpoint.get("trainable_state_dict", checkpoint.get("model_state_dict", checkpoint))
        encoder_state = {
            key.removeprefix("mri_encoder."): value
            for key, value in state.items()
            if key.startswith("mri_encoder.")
        }
        if not encoder_state:
            raise ValueError("CLARITY checkpoint contains no mri_encoder parameters")
        incompatible = encoder.load_state_dict(encoder_state, strict=False)
        loaded_lora_tensors = len(encoder_state) - len(incompatible.unexpected_keys)
        if loaded_lora_tensors == 0:
            raise ValueError("No CLARITY MRI encoder tensors matched the BrainIAC adapter")

    device = resolve_device(device_name)
    encoder.to(device).eval()
    loader = dataset_module.MRIVolumeLoader(str(mri_root))
    with Path(timeline_path).open("r", encoding="utf-8") as handle:
        patients = json.load(handle)["patients"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    skipped_existing = 0
    skipped_missing_mri = 0
    for patient_id, patient in sorted(patients.items()):
        timeline = sorted(patient.get("timeline", []), key=lambda item: float(item.get("mri_day", 0)))
        for timepoint in timeline:
            identifier = f"{patient_id}_Timepoint_{int(''.join(filter(str.isdigit, str(timepoint['tp_id']))))}"
            output_path = output_dir / latent_filename(patient_id, timepoint["tp_id"])
            if output_path.exists():
                skipped_existing += 1
                continue
            if not loader.has(identifier):
                skipped_missing_mri += 1
                continue
            volume = loader.load(identifier).unsqueeze(0).to(device)
            with torch.inference_mode():
                tokens = encoder(volume).squeeze(0)
            if output_kind == "mean":
                value = tokens.mean(dim=0)
            elif output_kind == "tokens":
                value = tokens
            else:
                raise ValueError("output_kind must be mean or tokens")
            np.save(output_path, value.float().cpu().numpy(), allow_pickle=False)
            extracted += 1
            print(f"extracted {identifier} -> {tuple(value.shape)}", flush=True)
            if limit is not None and extracted >= limit:
                break
        if limit is not None and extracted >= limit:
            break
    summary = {
        "encoder": "brainiac",
        "brainiac_checkpoint": str(Path(brainiac_checkpoint).resolve()),
        "clarity_checkpoint": str(Path(clarity_checkpoint).resolve()) if clarity_checkpoint else None,
        "timeline": str(Path(timeline_path).resolve()),
        "mri_root": str(Path(mri_root).resolve()),
        "tokens_per_modality": tokens_per_modality,
        "output_kind": output_kind,
        "output_dim": 768,
        "loaded_clarity_encoder_tensors": loaded_lora_tensors,
        "extracted": extracted,
        "skipped_existing": skipped_existing,
        "skipped_missing_mri": skipped_missing_mri,
    }
    with (output_dir / "extraction_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary

