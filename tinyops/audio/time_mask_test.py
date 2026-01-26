import numpy as np
import pytest
from tinygrad import Tensor

from tinyops._core import assert_close, assert_one_kernel
from tinyops.audio.time_mask import time_mask


@assert_one_kernel
def test_time_mask_basic(monkeypatch):
    spectrogram = Tensor.ones(1, 10, 20)
    # Mock rand to ensure some masking happens
    mock_values = [Tensor([0.5]), Tensor([0.2])]
    monkeypatch.setattr(Tensor, "rand", lambda *shape: mock_values.pop(0).expand(shape))

    masked = time_mask(spectrogram, time_mask_param=5, mask_value=0.0)
    assert masked.shape == (1, 10, 20)
    assert masked.sum().item() < spectrogram.sum().item()


@assert_one_kernel
def test_time_mask_deterministic(monkeypatch):
    spectrogram = Tensor.ones(1, 5, 10).realize()

    # Mock Tensor.rand
    mock_values = [Tensor([0.5]).reshape(1, 1, 1), Tensor([0.2]).reshape(1, 1, 1)]
    monkeypatch.setattr(Tensor, "rand", lambda *shape: mock_values.pop(0).expand(shape))

    # time_mask_param = 4
    # t = floor(0.5 * 4) = 2
    # t0 = floor(0.2 * (10 - 2)) = floor(1.6) = 1
    # Mask will be from time 1 up to (1+2)=3, so columns 1 and 2 are masked

    masked = time_mask(spectrogram, time_mask_param=4, mask_value=0.0).realize()

    expected_np = np.ones((1, 5, 10), dtype=np.float32)
    expected_np[0, :, 1:3] = 0.0
    expected = Tensor(expected_np)

    assert_close(masked, expected)


@assert_one_kernel
def test_time_mask_iid(monkeypatch):
    spectrogram = Tensor.ones(2, 5, 10).realize()

    # Mock Tensor.rand
    mock_values = [Tensor([[0.5], [0.8]]).reshape(2, 1, 1), Tensor([[0.2], [0.1]]).reshape(2, 1, 1)]
    monkeypatch.setattr(Tensor, "rand", lambda *shape: mock_values.pop(0).expand(shape))

    # Batch 1: t=2, t0=1
    # Batch 2: t=3, t0=0

    masked = time_mask(spectrogram, time_mask_param=4, mask_value=0.0, iid_masks=True).realize()

    expected_np = np.ones((2, 5, 10), dtype=np.float32)
    expected_np[0, :, 1:3] = 0.0
    expected_np[1, :, 0:3] = 0.0
    expected = Tensor(expected_np)

    assert_close(masked, expected)


@assert_one_kernel
def test_time_mask_param_full(monkeypatch):
    spectrogram = Tensor.ones(1, 5, 10).realize()
    mock_values = [Tensor([0.99]), Tensor([0.1])]
    monkeypatch.setattr(Tensor, "rand", lambda *shape: mock_values.pop(0).expand(shape))
    masked = time_mask(spectrogram, time_mask_param=10, mask_value=0.0).realize()
    assert masked.sum().item() == 5.0


@assert_one_kernel
def test_time_mask_param_zero():
    spectrogram = Tensor.ones(1, 5, 10).realize()
    masked = time_mask(spectrogram, time_mask_param=0, mask_value=0.0).realize()
    assert_close(masked, spectrogram)


def test_invalid_param():
    spectrogram = Tensor.ones(1, 5, 10)
    with pytest.raises(ValueError):
        time_mask(spectrogram, time_mask_param=-1)
    with pytest.raises(ValueError):
        time_mask(spectrogram, time_mask_param=11)
    with pytest.raises(ValueError):
        time_mask(Tensor.rand(5), time_mask_param=1)
