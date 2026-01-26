import math

from tinygrad import dtypes
from tinygrad.tensor import Tensor


def _fade_in(waveform_length: int, fade_in_len: int, fade_shape: str) -> Tensor:
    if fade_in_len <= 0:
        return Tensor.ones(waveform_length, dtype=dtypes.float32)
    fade = Tensor.linspace(0, 1, fade_in_len, dtype=dtypes.float32)
    ones = Tensor.ones(waveform_length - fade_in_len, dtype=dtypes.float32)

    if fade_shape == "linear":
        fade_curve = fade
    elif fade_shape == "exponential":
        fade_curve = Tensor(2, dtype=dtypes.float32).pow(fade - 1) * fade
    elif fade_shape == "logarithmic":
        fade_curve = (Tensor(0.1, dtype=dtypes.float32) + fade).log() / math.log(10) + 1
    elif fade_shape == "quarter_sine":
        fade_curve = (fade * math.pi / 2).sin()
    elif fade_shape == "half_sine":
        fade_curve = (fade * math.pi - math.pi / 2).sin() / 2 + 0.5
    else:
        raise ValueError(f"Unknown fade_shape for fade-in: {fade_shape}")
    return Tensor.cat(fade_curve, ones).clip(0, 1)


def _fade_out(waveform_length: int, fade_out_len: int, fade_shape: str) -> Tensor:
    if fade_out_len <= 0:
        return Tensor.ones(waveform_length, dtype=dtypes.float32)
    fade = Tensor.linspace(0, 1, fade_out_len, dtype=dtypes.float32)
    ones = Tensor.ones(waveform_length - fade_out_len, dtype=dtypes.float32)

    if fade_shape == "linear":
        fade_curve = -fade + 1
    elif fade_shape == "exponential":
        fade_curve = Tensor(2, dtype=dtypes.float32).pow(-fade) * (1 - fade)
    elif fade_shape == "logarithmic":
        fade_curve = (Tensor(1.1, dtype=dtypes.float32) - fade).log() / math.log(10) + 1
    elif fade_shape == "quarter_sine":
        fade_curve = (fade * math.pi / 2 + math.pi / 2).sin()
    elif fade_shape == "half_sine":
        fade_curve = (fade * math.pi + math.pi / 2).sin() / 2 + 0.5
    else:
        raise ValueError(f"Unknown fade_shape for fade-out: {fade_shape}")
    return Tensor.cat(ones, fade_curve).clip(0, 1)


def fade(waveform: Tensor, fade_in_len: int = 0, fade_out_len: int = 0, fade_shape: str = "linear") -> Tensor:
    """
    Add a fade in and/or fade out to a waveform. This function is a tinygrad implementation of
    torchaudio.transforms.Fade.

    Args:
        waveform (Tensor): Tensor of audio of dimension `(..., time)`.
        fade_in_len (int, optional): Length of fade-in (time frames). (Default: 0)
        fade_out_len (int, optional): Length of fade-out (time frames). (Default: 0)
        fade_shape (str, optional): Shape of fade. Must be one of: "quarter_sine",
            "half_sine", "linear", "logarithmic", "exponential". (Default: "linear")

    Returns:
        Tensor: Tensor of audio of dimension `(..., time)`.
    """
    waveform_length = waveform.shape[-1]
    if fade_in_len < 0 or fade_out_len < 0:
        raise ValueError("Fade length cannot be negative.")
    if fade_in_len > waveform_length or fade_out_len > waveform_length:
        raise ValueError("Fade length cannot be greater than waveform length.")
    fade_in_multiplier = _fade_in(waveform_length, fade_in_len, fade_shape)
    fade_out_multiplier = _fade_out(waveform_length, fade_out_len, fade_shape)

    return waveform * fade_in_multiplier * fade_out_multiplier
