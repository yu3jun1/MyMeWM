from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from .data import load_dataset, split_patient_ids
from .mri_core_extract import MODALITIES, _volume_path, preprocess_slices, select_slice_indices


@dataclass(frozen=True)
class Transition:
    patient_id: str
    source_timepoint: str
    target_timepoint: str
    action: np.ndarray
    delta_days: float
    frozen_source: np.ndarray


def timepoint_number(value: object) -> int:
    match = re.search(r"(\d+)", str(value))
    if match is None:
        raise ValueError(f"Cannot parse MRI timepoint number from {value!r}")
    return int(match.group(1))


class FrozenMRICoreSource:
    """Raw MRI and aligned frozen features for one fixed Stage 1 patient split."""

    def __init__(self, data_dir: str | Path, split_seed: int = 17,
                 train_fraction: float = 0.7,
                 validation_fraction: float = 0.15) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.trajectories, self.metadata = load_dataset(self.data_dir)
        provenance = self.metadata.get("provenance") or {}
        extraction = provenance.get("latent_extraction") or {}
        if provenance.get("action_anchor") != "source":
            raise ValueError("LoRA adaptation requires source-anchored treatments")
        if (extraction.get("encoder") != "mri_core" or
                extraction.get("adapter") is not None or
                extraction.get("output_kind") != "mean" or
                self.metadata["latent_dim"] != 256):
            raise ValueError("Expected the frozen MRI-CORE 256-dimensional mean-pooled dataset")
        self.extraction = extraction
        self.latent_dir = Path(provenance["latent_dir"]).resolve()
        self.mri_root = Path(extraction["mri_root"]).resolve()
        self.repository = Path(extraction["encoder_repository"]).resolve()
        self.checkpoint = Path(extraction["encoder_checkpoint"]).resolve()
        self.sam_checkpoint = Path(extraction["sam_checkpoint"]).resolve()
        self.timeline = Path(provenance["timeline"]).resolve()
        for path in (self.latent_dir, self.mri_root, self.repository,
                     self.checkpoint, self.sam_checkpoint, self.timeline):
            if not path.exists():
                raise FileNotFoundError(f"MRI-CORE input is missing: {path}")
        self.split_seed = split_seed
        self.split = split_patient_ids(
            (trajectory.patient_id for trajectory in self.trajectories),
            split_seed, train_fraction, validation_fraction)
        self.by_patient = {trajectory.patient_id: trajectory
                           for trajectory in self.trajectories}

    def assert_preprocessing(self, image_size: int, normalization: str,
                             slice_policy: str, slices_per_modality: int) -> None:
        expected = {
            "image_size": image_size,
            "normalization": normalization,
            "slice_policy": slice_policy,
            "slices_per_modality": (
                slices_per_modality if slice_policy == "uniform" else None),
        }
        mismatches = {
            key: (self.extraction.get(key), value)
            for key, value in expected.items()
            if self.extraction.get(key) != value
        }
        if mismatches:
            raise ValueError(f"LoRA preprocessing differs from frozen MRI-CORE: {mismatches}")

    def transitions(self, split_name: str) -> list[Transition]:
        if split_name not in self.split:
            raise ValueError(f"Unknown patient split: {split_name}")
        records = []
        for patient_id in self.split[split_name]:
            trajectory = self.by_patient[patient_id]
            for index in range(len(trajectory.latents) - 1):
                records.append(Transition(
                    patient_id=patient_id,
                    source_timepoint=str(trajectory.timepoints[index]),
                    target_timepoint=str(trajectory.timepoints[index + 1]),
                    action=trajectory.actions[index].copy(),
                    delta_days=float(trajectory.delta_days[index]),
                    frozen_source=trajectory.latents[index].copy(),
                ))
        return records

    def frozen_latent_files(self) -> list[Path]:
        files = sorted(self.latent_dir.glob("*_Timepoint_*.npy"))
        expected = self.extraction.get("extracted")
        if expected is not None and len(files) != expected:
            raise ValueError(
                f"Frozen latent file count differs from extraction metadata: "
                f"{len(files)} versus {expected}")
        return files

    def iter_preprocessed_slices(
        self, patient_id: str, timepoint: object, slice_batch_size: int,
        image_size: int, normalization: str,
        slice_policy: str, slices_per_modality: int,
    ) -> Iterator[torch.Tensor]:
        import nibabel as nib

        if slice_batch_size < 1:
            raise ValueError("slice_batch_size must be positive")
        number = timepoint_number(timepoint)
        for modality in MODALITIES:
            path = _volume_path(self.mri_root, patient_id, number, modality)
            if not path.is_file():
                raise FileNotFoundError(f"MRI volume is missing: {path}")
            volume = np.nan_to_num(nib.load(str(path)).get_fdata(dtype=np.float32))
            if volume.ndim != 3:
                raise ValueError(f"Expected 3D MRI volume at {path}, got {volume.shape}")
            indices = select_slice_indices(
                volume.shape[2], slice_policy, slices_per_modality)
            for start in range(0, len(indices), slice_batch_size):
                selected = indices[start:start + slice_batch_size]
                batch = np.moveaxis(volume[:, :, selected], 2, 0)
                yield preprocess_slices(batch, image_size, normalization)


def lora_output_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if not root.name.startswith("mri_core_lora"):
        raise ValueError(
            "Supplementary experiment output must have a distinct "
            "mri_core_lora* directory name"
        )
    return root
