from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from .clarity_adapter import latent_filename
from .training import resolve_device


MODALITIES = ("t1c", "t2w", "t1n", "t2f")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def select_slice_indices(depth: int, policy: str, slices_per_modality: int) -> np.ndarray:
    if depth < 1:
        raise ValueError("MRI volume has no slices")
    if policy == "all":
        return np.arange(depth, dtype=np.int64)
    if policy == "uniform":
        count = min(depth, slices_per_modality)
        if count < 1:
            raise ValueError("slices_per_modality must be positive")
        return np.unique(np.linspace(0, depth - 1, count).round().astype(np.int64))
    raise ValueError("slice_policy must be all or uniform")


def preprocess_slices(
    slices: np.ndarray,
    image_size: int,
    normalization: str,
) -> torch.Tensor:
    """Convert [B,H,W] MRI slices to MRI-CORE input [B,3,S,S]."""
    if slices.ndim != 3:
        raise ValueError(f"Expected [B,H,W] slices, got {slices.shape}")
    value = torch.from_numpy(np.nan_to_num(slices).astype(np.float32)).unsqueeze(1)
    flat = value.flatten(1)
    minimum = flat.min(dim=1).values[:, None, None, None]
    maximum = flat.max(dim=1).values[:, None, None, None]
    value = (value - minimum) / (maximum - minimum).clamp_min(1e-6)
    value = F.interpolate(value, size=(image_size, image_size), mode="bilinear", align_corners=False)
    value = value.repeat(1, 3, 1, 1)
    if normalization == "sam":
        mean = value.new_tensor(IMAGENET_MEAN)[None, :, None, None]
        std = value.new_tensor(IMAGENET_STD)[None, :, None, None]
        value = (value - mean) / std
    elif normalization != "minmax":
        raise ValueError("normalization must be minmax or sam")
    return value


def load_mri_core_image_encoder(
    mri_core_root: str | Path,
    checkpoint_path: str | Path,
    image_size: int,
) -> torch.nn.Module:
    mri_core_root = Path(mri_core_root).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    if not (mri_core_root / "models" / "sam").is_dir():
        raise FileNotFoundError(f"MRI-CORE repository not found at {mri_core_root}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"MRI-CORE checkpoint not found: {checkpoint_path}")
    if not mri_core_root.name.isidentifier():
        raise ValueError(
            "MRI-CORE repository directory name must be a valid Python identifier "
            f"(for example mri_foundation), got {mri_core_root.name!r}"
        )
    repository_parent = str(mri_core_root.parent)
    if repository_parent not in sys.path:
        sys.path.insert(0, repository_parent)
    module_name = f"{mri_core_root.name}.models.sam"
    sam_module = importlib.import_module(module_name)
    expected_module = (mri_core_root / "models" / "sam" / "__init__.py").resolve()
    if Path(sam_module.__file__).resolve() != expected_module:
        raise RuntimeError(
            f"{module_name} resolved outside the requested MRI-CORE repository: "
            f"{sam_module.__file__}"
        )
    args = SimpleNamespace(
        arch="vit_b",
        if_encoder_adapter=False,
        encoder_adapter_depths=[],
        if_mask_decoder_adapter=False,
        decoder_adapt_depth=0,
    )
    model = sam_module.sam_model_registry["vit_b"](
        args,
        checkpoint=str(checkpoint_path),
        num_classes=1,
        image_size=image_size,
        pretrained_sam=False,
    )
    encoder = model.image_encoder
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def _volume_path(mri_root: Path, patient_id: str, timepoint_number: int, modality: str) -> Path:
    return (
        mri_root
        / patient_id
        / f"Timepoint_{timepoint_number}"
        / f"{patient_id}_Timepoint_{timepoint_number}_brain_{modality}.nii.gz"
    )


def _encode_volume(
    image_encoder: torch.nn.Module,
    volume: np.ndarray,
    device: torch.device,
    image_size: int,
    normalization: str,
    slice_policy: str,
    slices_per_modality: int,
    slice_batch_size: int,
) -> torch.Tensor:
    indices = select_slice_indices(volume.shape[2], slice_policy, slices_per_modality)
    tokens = []
    for start in range(0, len(indices), slice_batch_size):
        batch_indices = indices[start : start + slice_batch_size]
        batch = np.moveaxis(volume[:, :, batch_indices], 2, 0)
        inputs = preprocess_slices(batch, image_size, normalization).to(device)
        with torch.inference_mode():
            feature_maps = image_encoder(inputs)
        if feature_maps.ndim != 4 or feature_maps.shape[1] != 256:
            raise ValueError(
                "MRI-CORE image_encoder must return [B,256,H,W], "
                f"got {tuple(feature_maps.shape)}"
            )
        tokens.append(feature_maps.mean(dim=(-2, -1)).float().cpu())
    return torch.cat(tokens, dim=0)


def extract_mri_core_latents(
    mri_core_root: str | Path,
    timeline_path: str | Path,
    mri_root: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
    image_size: int = 1024,
    normalization: str = "minmax",
    slice_policy: str = "all",
    slices_per_modality: int = 16,
    slice_batch_size: int = 2,
    output_kind: str = "mean",
) -> dict:
    import nibabel as nib

    if slice_batch_size < 1:
        raise ValueError("slice_batch_size must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Extraction output directory must be empty: {output_dir}")
    device = resolve_device(device_name)
    image_encoder = load_mri_core_image_encoder(mri_core_root, checkpoint_path, image_size).to(device)
    timeline_path = Path(timeline_path)
    mri_root = Path(mri_root)
    with timeline_path.open("r", encoding="utf-8") as handle:
        patients = json.load(handle)["patients"]

    extracted = 0
    skipped_missing_mri = 0
    for patient_id, patient in sorted(patients.items()):
        timeline = sorted(patient.get("timeline", []), key=lambda item: float(item.get("mri_day", 0)))
        for timepoint in timeline:
            digits = "".join(filter(str.isdigit, str(timepoint["tp_id"])))
            if not digits:
                raise ValueError(f"Cannot parse timepoint: {timepoint['tp_id']}")
            timepoint_number = int(digits)
            output_path = output_dir / latent_filename(patient_id, timepoint["tp_id"])
            paths = [_volume_path(mri_root, patient_id, timepoint_number, modality) for modality in MODALITIES]
            if not all(path.is_file() for path in paths):
                skipped_missing_mri += 1
                continue
            modality_tokens = []
            for path in paths:
                volume = np.nan_to_num(nib.load(str(path)).get_fdata(dtype=np.float32))
                modality_tokens.append(
                    _encode_volume(
                        image_encoder,
                        volume,
                        device,
                        image_size,
                        normalization,
                        slice_policy,
                        slices_per_modality,
                        slice_batch_size,
                    )
                )
            tokens = torch.cat(modality_tokens, dim=0)
            if output_kind == "mean":
                value = tokens.mean(dim=0)
            elif output_kind == "tokens":
                value = tokens
            else:
                raise ValueError("output_kind must be mean or tokens")
            np.save(output_path, value.numpy(), allow_pickle=False)
            extracted += 1
            print(f"extracted MRI-CORE {patient_id} T{timepoint_number} -> {tuple(value.shape)}", flush=True)
    summary = {
        "encoder": "mri_core",
        "frozen": True,
        "adapter": None,
        "encoder_repository": str(Path(mri_core_root).resolve()),
        "encoder_checkpoint": str(Path(checkpoint_path).resolve()),
        "timeline": str(timeline_path.resolve()),
        "mri_root": str(mri_root.resolve()),
        "image_size": image_size,
        "normalization": normalization,
        "slice_policy": slice_policy,
        "slices_per_modality": slices_per_modality if slice_policy == "uniform" else None,
        "slice_batch_size": slice_batch_size,
        "output_kind": output_kind,
        "output_dim": 256,
        "extracted": extracted,
        "skipped_missing_mri": skipped_missing_mri,
    }
    with (output_dir / "extraction_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
