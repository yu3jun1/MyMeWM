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
    device_name: str = "auto",
    tokens_per_modality: int = 8,
    output_kind: str = "mean",
) -> dict:
    clarity_root = Path(clarity_root).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Extraction output directory must be empty: {output_dir}")
    predictor_root = clarity_root / "Predictor"
    if str(predictor_root) not in sys.path:
        sys.path.insert(0, str(predictor_root))
    if str(clarity_root) not in sys.path:
        sys.path.insert(0, str(clarity_root))
    brainiac_module = importlib.import_module("models.brainiac_adapter")
    vision_module = importlib.import_module("models.vision_backbone")
    dataset_module = importlib.import_module("dataset.dataset_glioma_all_pairs_text")

    backbone = brainiac_module.BrainIACAdapter(
        checkpoint_path=str(brainiac_checkpoint),
        tokens_per_modality=tokens_per_modality,
        lora_r=0,
    )
    encoder = vision_module.MultiModalVisionBackbone(backbone, num_modalities=4)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    device = resolve_device(device_name)
    encoder.to(device).eval()
    loader = dataset_module.MRIVolumeLoader(str(mri_root))
    with Path(timeline_path).open("r", encoding="utf-8") as handle:
        patients = json.load(handle)["patients"]
    extracted = 0
    skipped_missing_mri = 0
    for patient_id, patient in sorted(patients.items()):
        timeline = sorted(patient.get("timeline", []), key=lambda item: float(item.get("mri_day", 0)))
        for timepoint in timeline:
            identifier = f"{patient_id}_Timepoint_{int(''.join(filter(str.isdigit, str(timepoint['tp_id']))))}"
            output_path = output_dir / latent_filename(patient_id, timepoint["tp_id"])
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
    summary = {
        "encoder": "brainiac",
        "brainiac_checkpoint": str(Path(brainiac_checkpoint).resolve()),
        "frozen": True,
        "adapter": None,
        "timeline": str(Path(timeline_path).resolve()),
        "mri_root": str(Path(mri_root).resolve()),
        "tokens_per_modality": tokens_per_modality,
        "output_kind": output_kind,
        "output_dim": 768,
        "extracted": extracted,
        "skipped_missing_mri": skipped_missing_mri,
    }
    with (output_dir / "extraction_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary

