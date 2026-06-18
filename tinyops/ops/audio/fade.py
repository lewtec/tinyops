import math
from enum import Enum

from tinygrad import Tensor, dtypes


class FadeShape(Enum):
    """Shape of the fade envelope."""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    QUARTER_SINE = "quarter_sine"
    HALF_SINE = "half_sine"


def _build_fade_in(waveform_length: int, fade_length: int, shape: FadeShape) -> Tensor:
    if fade_length <= 0:
        return Tensor.ones(waveform_length, dtype=dtypes.float32)

    positions = Tensor.linspace(0, 1, fade_length, dtype=dtypes.float32)
    ones = Tensor.ones(waveform_length - fade_length, dtype=dtypes.float32)

    if shape == FadeShape.LINEAR:
        curve = positions
    elif shape == FadeShape.EXPONENTIAL:
        curve = Tensor(2, dtype=dtypes.float32).pow(positions - 1) * positions
    elif shape == FadeShape.LOGARITHMIC:
        curve = (Tensor(0.1, dtype=dtypes.float32) + positions).log() / math.log(10) + 1
    elif shape == FadeShape.QUARTER_SINE:
        curve = (positions * math.pi / 2).sin()
    elif shape == FadeShape.HALF_SINE:
        curve = (positions * math.pi - math.pi / 2).sin() / 2 + 0.5
    else:
        raise ValueError(f"Unknown fade shape: {shape}")

    return Tensor.cat(curve, ones).clip(0, 1)


def _build_fade_out(waveform_length: int, fade_length: int, shape: FadeShape) -> Tensor:
    if fade_length <= 0:
        return Tensor.ones(waveform_length, dtype=dtypes.float32)

    positions = Tensor.linspace(0, 1, fade_length, dtype=dtypes.float32)
    ones = Tensor.ones(waveform_length - fade_length, dtype=dtypes.float32)

    if shape == FadeShape.LINEAR:
        curve = -positions + 1
    elif shape == FadeShape.EXPONENTIAL:
        curve = Tensor(2, dtype=dtypes.float32).pow(-positions) * (1 - positions)
    elif shape == FadeShape.LOGARITHMIC:
        curve = (Tensor(1.1, dtype=dtypes.float32) - positions).log() / math.log(10) + 1
    elif shape == FadeShape.QUARTER_SINE:
        curve = (positions * math.pi / 2 + math.pi / 2).sin()
    elif shape == FadeShape.HALF_SINE:
        curve = (positions * math.pi + math.pi / 2).sin() / 2 + 0.5
    else:
        raise ValueError(f"Unknown fade shape: {shape}")

    return Tensor.cat(ones, curve).clip(0, 1)


def fade(
    waveform: Tensor,
    fade_in_length: int = 0,
    fade_out_length: int = 0,
    shape: FadeShape = FadeShape.LINEAR,
) -> Tensor:
    """Apply fade-in and/or fade-out to a waveform.

    Args:
        waveform: Audio tensor of shape (..., time).
        fade_in_length: Number of frames for fade-in.
        fade_out_length: Number of frames for fade-out.
        shape: Envelope shape for the fade.

    Returns:
        Faded waveform tensor.

    Raises:
        ValueError: If fade lengths are negative or exceed waveform length.
    """
    waveform_length = waveform.shape[-1]
    if fade_in_length < 0 or fade_out_length < 0:
        raise ValueError("Fade length cannot be negative.")
    if fade_in_length > waveform_length or fade_out_length > waveform_length:
        raise ValueError("Fade length cannot be greater than waveform length.")

    fade_in_envelope = _build_fade_in(waveform_length, fade_in_length, shape)
    fade_out_envelope = _build_fade_out(waveform_length, fade_out_length, shape)

    return waveform * fade_in_envelope * fade_out_envelope
