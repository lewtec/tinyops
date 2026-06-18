import math
from enum import Enum

from tinygrad import Tensor


class SpectrogramScale(Enum):
    """Scale of the input spectrogram."""

    POWER = "power"
    MAGNITUDE = "magnitude"


_SCALE_MULTIPLIERS = {
    SpectrogramScale.POWER: 10.0,
    SpectrogramScale.MAGNITUDE: 20.0,
}


def amplitude_to_decibels(
    spectrogram: Tensor,
    scale: SpectrogramScale = SpectrogramScale.POWER,
    reference: float = 1.0,
    minimum_amplitude: float = 1e-10,
    dynamic_range: float | None = 80.0,
) -> Tensor:
    """Convert a spectrogram from amplitude/power scale to decibels.

    Args:
        spectrogram: Input spectrogram tensor.
        scale: Whether the input is in power or magnitude scale.
        reference: Reference value for dB computation.
        minimum_amplitude: Floor value to avoid log(0).
        dynamic_range: If not None, clamp output to max(output) - dynamic_range.

    Returns:
        Spectrogram in decibel scale.

    Raises:
        ValueError: If dynamic_range is negative.
    """
    multiplier = _SCALE_MULTIPLIERS[scale]
    reference_value = abs(reference)

    clamped = spectrogram.maximum(minimum_amplitude)
    decibels = multiplier * (clamped / reference_value).log() / math.log(10)

    if dynamic_range is not None:
        if dynamic_range < 0:
            raise ValueError("dynamic_range must be non-negative")
        maximum = decibels.max()
        decibels = decibels.maximum(maximum - dynamic_range)

    return decibels
