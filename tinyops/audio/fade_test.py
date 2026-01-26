import numpy as np
import pytest
import torch
import torchaudio.transforms as T
from tinygrad import Tensor

from tinyops._core import assert_close
from tinyops.audio.fade import fade


@pytest.mark.parametrize(
    "shape, fade_in_len, fade_out_len, fade_shape",
    [
        ((1, 44100), 4410, 8820, "linear"),
        ((2, 22050), 0, 5000, "quarter_sine"),
        ((1, 16000), 1600, 0, "half_sine"),
        ((1, 8000), 1000, 1000, "logarithmic"),
        ((2, 48000), 2400, 2400, "exponential"),
        ((1, 44100), 44100, 44100, "linear"),
    ],
)
def test_fade_parity(shape, fade_in_len, fade_out_len, fade_shape):
    np_waveform = np.random.randn(*shape).astype(np.float32)
    tinygrad_waveform = Tensor(np_waveform)
    torch_waveform = torch.from_numpy(np_waveform)

    tinygrad_faded = fade(tinygrad_waveform, fade_in_len, fade_out_len, fade_shape)

    torchaudio_fade = T.Fade(fade_in_len, fade_out_len, fade_shape)
    torch_faded = torchaudio_fade(torch_waveform)

    assert_close(tinygrad_faded, torch_faded.numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "shape, fade_in_len, fade_out_len",
    [
        (
            (1, 100),
            -1,
            10,
        ),
        (
            (1, 100),
            10,
            -1,
        ),
        (
            (1, 100),
            101,
            10,
        ),
        (
            (1, 100),
            10,
            101,
        ),
    ],
)
def test_fade_invalid_params(shape, fade_in_len, fade_out_len):
    np_waveform = np.random.randn(*shape).astype(np.float32)
    tinygrad_waveform = Tensor(np_waveform)
    with pytest.raises(ValueError):
        fade(tinygrad_waveform, fade_in_len, fade_out_len)
