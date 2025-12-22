import pytest
import numpy as np
from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.audio.freq_mask import freq_mask
from tinyops.test_utils import assert_one_kernel

# Pre-compute tensors for tests to avoid kernel counting issues
def _get_basic_inputs():
    spec = Tensor(np.ones((1, 10, 20), dtype=np.float32)).realize()
    r1 = Tensor(np.full((1, 1, 1), 0.5, dtype=np.float32)).realize()
    r2 = Tensor(np.full((1, 1, 1), 0.2, dtype=np.float32)).realize()
    f_idx = Tensor(np.arange(10, dtype=np.float32).reshape(1, 10, 1)).realize()
    return spec, (r1, r2), f_idx

BASIC_INPUTS = _get_basic_inputs()

@pytest.mark.parametrize("spectrogram, rand_values, freq_indices", [BASIC_INPUTS])
@assert_one_kernel
def test_freq_mask_basic(spectrogram, rand_values, freq_indices):
    masked = freq_mask(spectrogram, freq_mask_param=5, mask_value=0.0, _rand_values=rand_values, _freq_indices=freq_indices)
    assert masked.shape == (1, 10, 20)
    assert masked.sum().item() < spectrogram.sum().item()

def _get_deterministic_inputs():
    spec = Tensor(np.ones((1, 10, 5), dtype=np.float32)).realize()
    # freq_mask_param = 4
    # f = floor(0.5 * 4) = 2
    # f0 = floor(0.2 * (10 - 2)) = floor(1.6) = 1
    # Mask rows 1 and 2
    r1 = Tensor(np.full((1, 1, 1), 0.5, dtype=np.float32)).realize()
    r2 = Tensor(np.full((1, 1, 1), 0.2, dtype=np.float32)).realize()
    f_idx = Tensor(np.arange(10, dtype=np.float32).reshape(1, 10, 1)).realize()

    expected_np = np.ones((1, 10, 5), dtype=np.float32)
    expected_np[0, 1:3, :] = 0.0
    expected = Tensor(expected_np).realize()
    return spec, (r1, r2), f_idx, expected

DETERMINISTIC_INPUTS = _get_deterministic_inputs()

@pytest.mark.parametrize("spectrogram, rand_values, freq_indices, expected", [DETERMINISTIC_INPUTS])
@assert_one_kernel
def test_freq_mask_deterministic(spectrogram, rand_values, freq_indices, expected):
    masked = freq_mask(spectrogram, freq_mask_param=4, mask_value=0.0, _rand_values=rand_values, _freq_indices=freq_indices)
    assert_close(masked, expected)

def _get_iid_inputs():
    spec = Tensor(np.ones((2, 10, 5), dtype=np.float32)).realize()
    # Batch 1: f=2, f0=1
    # Batch 2: f=3, f0=0
    r1 = Tensor(np.array([[[0.5]], [[0.8]]], dtype=np.float32)).realize()
    r2 = Tensor(np.array([[[0.2]], [[0.1]]], dtype=np.float32)).realize()
    f_idx = Tensor(np.arange(10, dtype=np.float32).reshape(1, 10, 1)).realize()

    expected_np = np.ones((2, 10, 5), dtype=np.float32)
    expected_np[0, 1:3, :] = 0.0
    expected_np[1, 0:3, :] = 0.0
    expected = Tensor(expected_np).realize()
    return spec, (r1, r2), f_idx, expected

IID_INPUTS = _get_iid_inputs()

@pytest.mark.parametrize("spectrogram, rand_values, freq_indices, expected", [IID_INPUTS])
@assert_one_kernel
def test_freq_mask_iid(spectrogram, rand_values, freq_indices, expected):
    masked = freq_mask(spectrogram, freq_mask_param=4, mask_value=0.0, iid_masks=True, _rand_values=rand_values, _freq_indices=freq_indices)
    assert_close(masked, expected)

def _get_param_full_inputs():
    spec = Tensor(np.ones((1, 10, 5), dtype=np.float32)).realize()
    r1 = Tensor(np.full((1, 1, 1), 0.99, dtype=np.float32)).realize()
    r2 = Tensor(np.full((1, 1, 1), 0.1, dtype=np.float32)).realize()
    f_idx = Tensor(np.arange(10, dtype=np.float32).reshape(1, 10, 1)).realize()
    return spec, (r1, r2), f_idx

PARAM_FULL_INPUTS = _get_param_full_inputs()

@pytest.mark.parametrize("spectrogram, rand_values, freq_indices", [PARAM_FULL_INPUTS])
@assert_one_kernel
def test_freq_mask_param_full(spectrogram, rand_values, freq_indices):
    masked = freq_mask(spectrogram, freq_mask_param=10, mask_value=0.0, _rand_values=rand_values, _freq_indices=freq_indices)
    # Only the last row should remain unmasked? Wait.
    # f = floor(0.99 * 10) = 9
    # f0 = floor(0.1 * (10 - 9)) = floor(0.1) = 0
    # Mask from 0 to 9. 10 rows (0-9). Mask indices [0, 1, ... 8].
    # Row 9 remains.
    # So sum should be 1 * 5 = 5.
    assert masked.sum().item() == 5.0

def _get_param_zero_inputs():
    spec = Tensor(np.ones((1, 10, 5), dtype=np.float32)).realize()
    return spec

PARAM_ZERO_INPUTS = _get_param_zero_inputs()

@pytest.mark.parametrize("spectrogram", [PARAM_ZERO_INPUTS])
@assert_one_kernel
def test_freq_mask_param_zero(spectrogram):
    masked = freq_mask(spectrogram, freq_mask_param=0, mask_value=0.0)
    assert_close(masked, spectrogram)

def test_invalid_param():
    spectrogram = Tensor.ones(1, 10, 5)
    with pytest.raises(ValueError):
        freq_mask(spectrogram, freq_mask_param=-1)
    with pytest.raises(ValueError):
        freq_mask(spectrogram, freq_mask_param=11)
    with pytest.raises(ValueError):
        freq_mask(Tensor.rand(5), freq_mask_param=1)
