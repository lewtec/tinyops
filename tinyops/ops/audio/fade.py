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


def _fade_shape_curve(positions: Tensor, shape: FadeShape) -> Tensor:
    """Map unit-interval positions to a fade-in amplitude curve."""
    if shape == FadeShape.LINEAR:
        return positions
    if shape == FadeShape.EXPONENTIAL:
        return Tensor(2, dtype=dtypes.float32).pow(positions - 1) * positions
    if shape == FadeShape.LOGARITHMIC:
        return (Tensor(0.1, dtype=dtypes.float32) + positions).log() / math.log(10) + 1
    if shape == FadeShape.QUARTER_SINE:
        return (positions * math.pi / 2).sin()
    if shape == FadeShape.HALF_SINE:
        return (positions * math.pi - math.pi / 2).sin() / 2 + 0.5
    raise ValueError(f"Unknown fade shape: {shape}")


def _build_fade_envelope(
    waveform_length: int,
    fade_length: int,
    shape: FadeShape,
    *,
    fade_out: bool,
) -> Tensor:
    if fade_length <= 0:
        return Tensor.ones(waveform_length, dtype=dtypes.float32)

    positions = Tensor.linspace(0, 1, fade_length, dtype=dtypes.float32)
    ones = Tensor.ones(waveform_length - fade_length, dtype=dtypes.float32)

    # Fade-out curves match fade-in evaluated at reversed positions (1 - t).
    curve_positions = (1 - positions) if fade_out else positions
    curve = _fade_shape_curve(curve_positions, shape).clip(0, 1)

    if fade_out:
        return Tensor.cat(ones, curve)
    return Tensor.cat(curve, ones)


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

    fade_in_envelope = _build_fade_envelope(waveform_length, fade_in_length, shape, fade_out=False)
    fade_out_envelope = _build_fade_envelope(waveform_length, fade_out_length, shape, fade_out=True)

    return waveform * fade_in_envelope * fade_out_envelope
