from tinygrad import Tensor


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
    if spectrogram.ndim < 2:
        raise ValueError("Input spectrogram must be at least 2D.")
    if maximum_mask_length < 0:
        raise ValueError("maximum_mask_length must be non-negative.")

    time_step_count = spectrogram.shape[-1]
    if maximum_mask_length > time_step_count:
        raise ValueError(f"maximum_mask_length ({maximum_mask_length}) must be <= time_step_count ({time_step_count})")

    if maximum_mask_length == 0:
        return spectrogram

    if independent_masks and spectrogram.ndim > 2:
        random_shape = spectrogram.shape[:-2] + (1, 1)
    else:
        random_shape = (1,) * spectrogram.ndim

    if _random_values is not None:
        random_length, random_start = _random_values
    else:
        random_length = Tensor.rand(*random_shape)
        random_start = Tensor.rand(*random_shape)

    mask_length = (random_length * maximum_mask_length).floor()
    mask_start = (random_start * (time_step_count - mask_length)).floor()

    if _time_indices is None:
        index_shape = [1] * spectrogram.ndim
        index_shape[-1] = time_step_count
        time_indices = Tensor.arange(time_step_count, dtype=spectrogram.dtype).reshape(index_shape)
    else:
        time_indices = _time_indices

    mask = (time_indices >= mask_start) & (time_indices < mask_start + mask_length)
    return spectrogram.where(mask.logical_not(), mask_value)
