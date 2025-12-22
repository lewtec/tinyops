import pytest
import numpy as np
from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.audio.time_mask import time_mask
from tinyops.test_utils import assert_one_kernel

def _get_basic_inputs():
    spec = Tensor(np.ones((1, 10, 20), dtype=np.float32)).realize()
    r1 = Tensor(np.full((1, 1, 1), 0.5, dtype=np.float32)).realize()
    r2 = Tensor(np.full((1, 1, 1), 0.2, dtype=np.float32)).realize()
    t_idx = Tensor(np.arange(20, dtype=np.float32).reshape(1, 1, 20)).realize()
    return spec, (r1, r2), t_idx

BASIC_INPUTS = _get_basic_inputs()

@pytest.mark.parametrize("spectrogram, rand_values, time_indices", [BASIC_INPUTS])
@assert_one_kernel
def test_time_mask_basic(spectrogram, rand_values, time_indices):
    masked = time_mask(spectrogram, time_mask_param=5, mask_value=0.0, _rand_values=rand_values, _time_indices=time_indices)
    assert masked.shape == (1, 10, 20)
    assert masked.sum().item() < spectrogram.sum().item()

def _get_deterministic_inputs():
    spec = Tensor(np.ones((1, 5, 10), dtype=np.float32)).realize()
    # time_mask_param = 4
    # t = floor(0.5 * 4) = 2
    # t0 = floor(0.2 * (10 - 2)) = floor(1.6) = 1
    # Mask cols 1 and 2
    r1 = Tensor(np.full((1, 1, 1), 0.5, dtype=np.float32)).realize()
    r2 = Tensor(np.full((1, 1, 1), 0.2, dtype=np.float32)).realize()
    t_idx = Tensor(np.arange(10, dtype=np.float32).reshape(1, 1, 10)).realize()

    expected_np = np.ones((1, 5, 10), dtype=np.float32)
    expected_np[0, :, 1:3] = 0.0
    expected = Tensor(expected_np).realize()
    return spec, (r1, r2), t_idx, expected

DETERMINISTIC_INPUTS = _get_deterministic_inputs()

@pytest.mark.parametrize("spectrogram, rand_values, time_indices, expected", [DETERMINISTIC_INPUTS])
@assert_one_kernel
def test_time_mask_deterministic(spectrogram, rand_values, time_indices, expected):
    masked = time_mask(spectrogram, time_mask_param=4, mask_value=0.0, _rand_values=rand_values, _time_indices=time_indices)
    assert_close(masked, expected)

def _get_iid_inputs():
    spec = Tensor(np.ones((2, 5, 10), dtype=np.float32)).realize()
    # Batch 1: t=2, t0=1
    # Batch 2: t=3, t0=0
    r1 = Tensor(np.array([[[0.5]], [[0.8]]], dtype=np.float32)).realize()
    r2 = Tensor(np.array([[[0.2]], [[0.1]]], dtype=np.float32)).realize()
    t_idx = Tensor(np.arange(10, dtype=np.float32).reshape(1, 1, 10)).realize()

    expected_np = np.ones((2, 5, 10), dtype=np.float32)
    expected_np[0, :, 1:3] = 0.0
    expected_np[1, :, 0:3] = 0.0
    expected = Tensor(expected_np).realize()
    return spec, (r1, r2), t_idx, expected

IID_INPUTS = _get_iid_inputs()

@pytest.mark.parametrize("spectrogram, rand_values, time_indices, expected", [IID_INPUTS])
@assert_one_kernel
def test_time_mask_iid(spectrogram, rand_values, time_indices, expected):
    masked = time_mask(spectrogram, time_mask_param=4, mask_value=0.0, iid_masks=True, _rand_values=rand_values, _time_indices=time_indices)
    assert_close(masked, expected)

def _get_param_full_inputs():
    spec = Tensor(np.ones((1, 5, 10), dtype=np.float32)).realize()
    r1 = Tensor(np.full((1, 1, 1), 0.99, dtype=np.float32)).realize()
    r2 = Tensor(np.full((1, 1, 1), 0.1, dtype=np.float32)).realize()
    t_idx = Tensor(np.arange(10, dtype=np.float32).reshape(1, 1, 10)).realize()
    return spec, (r1, r2), t_idx

PARAM_FULL_INPUTS = _get_param_full_inputs()

@pytest.mark.parametrize("spectrogram, rand_values, time_indices", [PARAM_FULL_INPUTS])
@assert_one_kernel
def test_time_mask_param_full(spectrogram, rand_values, time_indices):
    masked = time_mask(spectrogram, time_mask_param=10, mask_value=0.0, _rand_values=rand_values, _time_indices=time_indices)
    # t = 9, t0=0. Mask cols 0-8. Col 9 remains.
    # sum = 5 * 1 = 5.0
    assert masked.sum().item() == 5.0

def _get_param_zero_inputs():
    spec = Tensor(np.ones((1, 5, 10), dtype=np.float32)).realize()
    return spec

PARAM_ZERO_INPUTS = _get_param_zero_inputs()

@pytest.mark.parametrize("spectrogram", [PARAM_ZERO_INPUTS])
@assert_one_kernel
def test_time_mask_param_zero(spectrogram):
    masked = time_mask(spectrogram, time_mask_param=0, mask_value=0.0)
    assert_close(masked, spectrogram)

def test_invalid_param():
    spectrogram = Tensor.ones(1, 5, 10)
    with pytest.raises(ValueError):
        time_mask(spectrogram, time_mask_param=-1)
    with pytest.raises(ValueError):
        time_mask(spectrogram, time_mask_param=11)
    with pytest.raises(ValueError):
        time_mask(Tensor.rand(5), time_mask_param=1)
