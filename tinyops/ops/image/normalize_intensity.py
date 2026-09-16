from tinygrad import Tensor, dtypes


def normalize_intensity(
    image: Tensor,
    target_minimum: float = 0,
    target_maximum: float = 255,
) -> Tensor:
    """Normalize image intensity to a target range using min-max scaling.

    Args:
        image: Input image tensor.
        target_minimum: Desired minimum output value.
        target_maximum: Desired maximum output value.

    Returns:
        Normalized image tensor.
    """
    source_minimum = image.min()
    source_maximum = image.max()

    dynamic_range = source_maximum - source_minimum
    is_constant = dynamic_range.eq(0)

    safe_range = is_constant.where(1.0, dynamic_range)
    scale = (target_maximum - target_minimum) / safe_range
    normalized = (image - source_minimum) * scale + target_minimum

    result = is_constant.where(
        Tensor.full(image.shape, target_minimum, dtype=image.dtype),
        normalized,
    )

    if image.dtype in (dtypes.uint8, dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64):
        return result.cast(image.dtype)
    return result
