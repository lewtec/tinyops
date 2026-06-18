from tinygrad import Tensor, dtypes

from tinyops.ops.image._filtering import apply_convolution_filter
from tinyops.ops.image.sobel_gradient import GradientDirection

_SCHARR_HORIZONTAL = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]
_SCHARR_VERTICAL = [[-3, -10, -3], [0, 0, 0], [3, 10, 3]]


def scharr_gradient(
    image: Tensor,
    direction: GradientDirection,
    scale: float = 1.0,
    delta: float = 0.0,
) -> Tensor:
    """Compute image gradient using the Scharr operator.

    The Scharr operator provides better rotational symmetry than Sobel.

    Args:
        image: Input image tensor.
        direction: Gradient direction (HORIZONTAL or VERTICAL).
        scale: Scale factor for output.
        delta: Value added to output.

    Returns:
        Gradient image tensor.
    """
    input_dtype = image.dtype
    compute_dtype = dtypes.float32 if input_dtype == dtypes.uint8 else input_dtype

    if direction == GradientDirection.HORIZONTAL:
        kernel = Tensor(_SCHARR_HORIZONTAL, dtype=compute_dtype)
    elif direction == GradientDirection.VERTICAL:
        kernel = Tensor(_SCHARR_VERTICAL, dtype=compute_dtype)
    else:
        raise ValueError(f"Invalid direction: {direction}")

    return apply_convolution_filter(image, kernel, scale, delta)
