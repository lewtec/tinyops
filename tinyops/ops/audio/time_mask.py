from tinygrad import Tensor

from tinyops.ops.audio._masking import _spectrogram_mask


def time_mask(
    spectrogram: Tensor,
    maximum_mask_length: int,
    mask_value: float = 0.0,
    independent_masks: bool = False,
    _random_values: tuple | None = None,
    _time_indices: Tensor | None = None,
) -> Tensor:
    """Mask a contiguous span of time steps in a spectrogram.

    Args:
        spectrogram: Input spectrogram of shape (..., frequency, time).
        maximum_mask_length: Maximum number of time steps to mask.
        mask_value: Value to fill in the masked region.
        independent_masks: If True, apply different masks per batch element.
        _random_values: Internal testing parameter.
        _time_indices: Internal testing parameter.

    Returns:
        Masked spectrogram tensor.
    """
    return _spectrogram_mask(
        spectrogram=spectrogram,
        maximum_mask_length=maximum_mask_length,
        axis=-1,
        mask_value=mask_value,
        independent_masks=independent_masks,
        _random_values=_random_values,
        _indices=_time_indices,
    )
