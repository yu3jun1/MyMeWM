import numpy as np
import pytest
import torch

from clarity_hauwm.encoder_comparison import parse_encoder_datasets, validate_encoder_alignment
from clarity_hauwm.mri_core_extract import (
    load_mri_core_image_encoder,
    preprocess_slices,
    select_slice_indices,
)


def test_mri_core_slice_preprocessing():
    assert np.array_equal(select_slice_indices(5, "all", 2), np.arange(5))
    indices = select_slice_indices(10, "uniform", 4)
    assert len(indices) == 4
    slices = np.stack([np.zeros((8, 9)), np.arange(72).reshape(8, 9)]).astype(np.float32)
    output = preprocess_slices(slices, image_size=16, normalization="minmax")
    assert output.shape == (2, 3, 16, 16)
    assert torch.isfinite(output).all()
    assert 0 <= output.min() <= output.max() <= 1


def test_encoder_dataset_alignment_and_parser(tmp_path, dataset_factory):
    first = tmp_path / "first"
    second = tmp_path / "second"
    dataset_factory(first, patients=6, seed=4)
    dataset_factory(second, patients=6, seed=4)
    datasets = parse_encoder_datasets([f"a={first}", f"b={second}"])
    result = validate_encoder_alignment(datasets)
    assert result["aligned"] is True
    assert result["patients"] == 6


def test_mri_core_requires_sam_checkpoint(tmp_path):
    repository = tmp_path / "mri_foundation"
    (repository / "models" / "sam").mkdir(parents=True)
    mri_checkpoint = tmp_path / "mri.pth"
    mri_checkpoint.touch()
    with pytest.raises(FileNotFoundError, match="SAM checkpoint"):
        load_mri_core_image_encoder(repository, mri_checkpoint, tmp_path / "sam.pth", 1024)
