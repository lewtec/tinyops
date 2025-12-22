import pytest
import numpy as np
from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.audio.freq_mask import freq_mask
from tinyops.test_utils import assert_one_kernel

# Fixtures para criar tensors fora do bloco medido
@pytest.fixture(scope="module")
def spectrogram_basic():
    return Tensor(np.ones((1, 10, 20), dtype=np.float32)).realize()

@pytest.fixture(scope="module")
def rand_values_basic():
    return (
        Tensor(np.full((1, 1, 1), 0.5, dtype=np.float32)),
        Tensor(np.full((1, 1, 1), 0.2, dtype=np.float32))
    )

@pytest.fixture(scope="module")
def freq_indices_10():
    return Tensor(np.arange(10, dtype=np.float32).reshape(1, 10, 1))

@assert_one_kernel
def test_freq_mask_basic(spectrogram_basic, rand_values_basic, freq_indices_10):
    rand1, rand2 = rand_values_basic
    masked = freq_mask(spectrogram_basic, freq_mask_param=5, mask_value=0.0, _rand_values=(rand1, rand2), _freq_indices=freq_indices_10)
    assert masked.shape == (1, 10, 20)
    assert masked.sum().item() < spectrogram_basic.sum().item()

@assert_one_kernel
def test_freq_mask_deterministic():
    spectrogram = Tensor(np.ones((1, 10, 5), dtype=np.float32)).realize()

    # freq_mask_param = 4
    # f = floor(0.5 * 4) = 2
    # f0 = floor(0.2 * (10 - 2)) = floor(1.6) = 1
    # Mask will be from freq 1 up to (1+2)=3, so rows 1 and 2 are masked
    rand1 = Tensor(np.full((1, 1, 1), 0.5, dtype=np.float32))
    rand2 = Tensor(np.full((1, 1, 1), 0.2, dtype=np.float32))
    freq_indices = Tensor(np.arange(10, dtype=np.float32).reshape(1, 10, 1))

    masked = freq_mask(spectrogram, freq_mask_param=4, mask_value=0.0, _rand_values=(rand1, rand2), _freq_indices=freq_indices)

    expected_np = np.ones((1, 10, 5), dtype=np.float32)
    expected_np[0, 1:3, :] = 0.0
    expected = Tensor(expected_np)

    assert_close(masked, expected)

@assert_one_kernel
def test_freq_mask_iid():
    spectrogram = Tensor(np.ones((2, 10, 5), dtype=np.float32)).realize()

    # Batch 1: f=2, f0=1 (same as before)
    # Batch 2:
    #   freq_mask_param = 4
    #   f = floor(0.8 * 4) = 3
    #   f0 = floor(0.1 * (10 - 3)) = floor(0.7) = 0
    #   Mask is from 0 up to 3, so rows 0, 1, 2 are masked
    rand1 = Tensor(np.array([[[0.5]], [[0.8]]], dtype=np.float32))
    rand2 = Tensor(np.array([[[0.2]], [[0.1]]], dtype=np.float32))
    freq_indices = Tensor(np.arange(10, dtype=np.float32).reshape(1, 10, 1))

    masked = freq_mask(spectrogram, freq_mask_param=4, mask_value=0.0, iid_masks=True, _rand_values=(rand1, rand2), _freq_indices=freq_indices)

    expected_np = np.ones((2, 10, 5), dtype=np.float32)
    expected_np[0, 1:3, :] = 0.0
    expected_np[1, 0:3, :] = 0.0
    expected = Tensor(expected_np)

    assert_close(masked, expected)

@assert_one_kernel
def test_freq_mask_param_full():
    spectrogram = Tensor(np.ones((1, 10, 5), dtype=np.float32)).realize()

    # f = floor(0.99 * 10) = 9
    # f0 = floor(0.1 * (10 - 9)) = floor(0.1) = 0
    rand1 = Tensor(np.full((1, 1, 1), 0.99, dtype=np.float32))
    rand2 = Tensor(np.full((1, 1, 1), 0.1, dtype=np.float32))
    freq_indices = Tensor(np.arange(10, dtype=np.float32).reshape(1, 10, 1))

    masked = freq_mask(spectrogram, freq_mask_param=10, mask_value=0.0, _rand_values=(rand1, rand2), _freq_indices=freq_indices)
    # Only the last row should remain unmasked. Shape is (1, 10, 5), so sum of one row is 5.
    assert masked.sum().item() == 5.0

@assert_one_kernel
def test_freq_mask_param_zero():
    spectrogram = Tensor(np.ones((1, 10, 5), dtype=np.float32)).realize()
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
