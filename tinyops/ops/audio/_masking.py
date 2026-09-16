from tinygrad import Tensor


def _spectrogram_mask(
    spectrogram: Tensor,
    maximum_mask_length: int,
    axis: int,
    mask_value: float = 0.0,
    independent_masks: bool = False,
    _random_values: tuple | None = None,
    _indices: Tensor | None = None,
) -> Tensor:
    """Mask a contiguous region along a specific axis in a spectrogram."""
    if spectrogram.ndim < 2:
        raise ValueError("Input spectrogram must be at least 2D.")
    if maximum_mask_length < 0:
        raise ValueError("maximum_mask_length must be non-negative.")

    dim_count = spectrogram.shape[axis]
    if maximum_mask_length > dim_count:
        raise ValueError(f"maximum_mask_length ({maximum_mask_length}) must be <= dimension size ({dim_count})")

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
    mask_start = (random_start * (dim_count - mask_length)).floor()

    if _indices is None:
        index_shape = [1] * spectrogram.ndim
        index_shape[axis] = dim_count
        indices = Tensor.arange(dim_count, dtype=spectrogram.dtype).reshape(index_shape)
    else:
        indices = _indices

    mask = (indices >= mask_start) & (indices < mask_start + mask_length)
    return spectrogram.where(mask.logical_not(), mask_value)
