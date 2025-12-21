import pytest
import numpy as np
from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.audio.freq_mask import freq_mask

def test_freq_mask_basic(monkeypatch):
    spectrogram = Tensor.ones(1, 10, 20)
    # Mock rand to ensure some masking happens, making the test deterministic
    mock_values = [Tensor([0.5]), Tensor([0.2])]
    monkeypatch.setattr(Tensor, "rand", lambda *shape: mock_values.pop(0).expand(shape))

    masked = freq_mask(spectrogram, freq_mask_param=5, mask_value=0.0)
    assert masked.shape == (1, 10, 20)
    assert masked.sum().item() < spectrogram.sum().item()

def test_freq_mask_deterministic(monkeypatch):
    spectrogram = Tensor.ones(1, 10, 5)

    # Mock Tensor.rand to return deterministic values
    mock_values = [Tensor([0.5]).reshape(1, 1, 1), Tensor([0.2]).reshape(1, 1, 1)]
    monkeypatch.setattr(Tensor, "rand", lambda *shape: mock_values.pop(0).expand(shape))

    # freq_mask_param = 4
    # f = floor(0.5 * 4) = 2
    # f0 = floor(0.2 * (10 - 2)) = floor(1.6) = 1
    # Mask will be from freq 1 up to (1+2)=3, so rows 1 and 2 are masked

    masked = freq_mask(spectrogram, freq_mask_param=4, mask_value=0.0)

    expected_np = np.ones((1, 10, 5), dtype=np.float32)
    expected_np[0, 1:3, :] = 0.0
    expected = Tensor(expected_np)

    assert_close(masked, expected)

def test_freq_mask_iid(monkeypatch):
    spectrogram = Tensor.ones(2, 10, 5)

    # Mock Tensor.rand to return deterministic values for a batch, ensuring correct shape
    mock_values = [Tensor([[0.5], [0.8]]).reshape(2, 1, 1), Tensor([[0.2], [0.1]]).reshape(2, 1, 1)]
    monkeypatch.setattr(Tensor, "rand", lambda *shape: mock_values.pop(0).expand(shape))

    # Batch 1: f=2, f0=1 (same as before)
    # Batch 2:
    #   freq_mask_param = 4
    #   f = floor(0.8 * 4) = 3
    #   f0 = floor(0.1 * (10 - 3)) = floor(0.7) = 0
    #   Mask is from 0 up to 3, so rows 0, 1, 2 are masked

    masked = freq_mask(spectrogram, freq_mask_param=4, mask_value=0.0, iid_masks=True)

    expected_np = np.ones((2, 10, 5), dtype=np.float32)
    expected_np[0, 1:3, :] = 0.0
    expected_np[1, 0:3, :] = 0.0
    expected = Tensor(expected_np)

    assert_close(masked, expected)

def test_freq_mask_param_full(monkeypatch):
    spectrogram = Tensor.ones(1, 10, 5)

    # Mock rand to create a mask of length 9 starting at 0.
    # f = floor(0.99 * 10) = 9
    # f0 = floor(0.1 * (10 - 9)) = floor(0.1) = 0
    mock_values = [Tensor([0.99]), Tensor([0.1])]
    monkeypatch.setattr(Tensor, "rand", lambda *shape: mock_values.pop(0).expand(shape))

    masked = freq_mask(spectrogram, freq_mask_param=10, mask_value=0.0)
    # Only the last row should remain unmasked. Shape is (1, 10, 5), so sum of one row is 5.
    assert masked.sum().item() == 5.0

def test_freq_mask_param_zero():
    spectrogram = Tensor.ones(1, 10, 5)
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
