import tempfile
from pathlib import Path

import numpy as np
import torch

from clarity_hauwm.encoder_comparison import parse_encoder_datasets, validate_encoder_alignment
from clarity_hauwm.mri_core_extract import preprocess_slices, select_slice_indices
from clarity_hauwm.synthetic import generate_synthetic_dataset


def test_mri_core_slice_preprocessing():
    assert np.array_equal(select_slice_indices(5, "all", 2), np.arange(5))
    indices = select_slice_indices(10, "uniform", 4)
    assert len(indices) == 4
    slices = np.stack([np.zeros((8, 9)), np.arange(72).reshape(8, 9)]).astype(np.float32)
    output = preprocess_slices(slices, image_size=16, normalization="minmax")
    assert output.shape == (2, 3, 16, 16)
    assert torch.isfinite(output).all()
    assert 0 <= output.min() <= output.max() <= 1


def test_encoder_dataset_alignment_and_parser():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first = root / "first"
        second = root / "second"
        generate_synthetic_dataset(first, patients=6, seed=4)
        generate_synthetic_dataset(second, patients=6, seed=4)
        datasets = parse_encoder_datasets([f"a={first}", f"b={second}"])
        result = validate_encoder_alignment(datasets)
        assert result["aligned"] is True
        assert result["patients"] == 6
