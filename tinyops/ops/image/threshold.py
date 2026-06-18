from enum import Enum

from tinygrad import Tensor


class ThresholdMethod(Enum):
    """Thresholding methods."""
    BINARY = "binary"
    BINARY_INVERSE = "binary_inverse"
    TRUNCATE = "truncate"
    TO_ZERO = "to_zero"
    TO_ZERO_INVERSE = "to_zero_inverse"


def apply_threshold(
    image: Tensor,
    threshold_value: float,
    maximum_value: float,
    method: ThresholdMethod,
) -> Tensor:
    """Apply a fixed-level threshold to a single-channel image.

    Args:
        image: Input image tensor.
        threshold_value: Threshold level.
        maximum_value: Maximum output value (used by BINARY methods).
        method: Thresholding strategy.

    Returns:
        Thresholded image tensor.
    """
    above_threshold = image > threshold_value

    if method == ThresholdMethod.BINARY:
        return above_threshold.where(maximum_value, 0)
    elif method == ThresholdMethod.BINARY_INVERSE:
        return above_threshold.where(0, maximum_value)
    elif method == ThresholdMethod.TRUNCATE:
        return above_threshold.where(threshold_value, image)
    elif method == ThresholdMethod.TO_ZERO:
        return above_threshold.where(image, 0)
    elif method == ThresholdMethod.TO_ZERO_INVERSE:
        return above_threshold.where(0, image)
    else:
        raise ValueError(f"Unknown threshold method: {method}")
