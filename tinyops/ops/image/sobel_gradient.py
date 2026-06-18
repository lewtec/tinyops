from enum import Enum

from tinygrad import Tensor, dtypes

from tinyops.ops.image._filtering import apply_convolution_filter
from tinyops.ops.image.pad import PaddingMode


class GradientDirection(Enum):
    """Direction for gradient computation."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


# Sobel kernels for 3x3
_SOBEL_HORIZONTAL_3x3 = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
_SOBEL_VERTICAL_3x3 = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]


def sobel_gradient(
    image: Tensor,
    direction: GradientDirection,
    kernel_size: int = 3,
    scale: float = 1.0,
    delta: float = 0.0,
) -> Tensor:
    """Compute image gradient using the Sobel operator.

    Args:
        image: Input image tensor.
        direction: Gradient direction (HORIZONTAL or VERTICAL).
        kernel_size: Size of the Sobel kernel (must be 3).
        scale: Scale factor for output.
        delta: Value added to output.

    Returns:
        Gradient image tensor.

    Raises:
        ValueError: If kernel_size is not 3 or direction is invalid.
    """
    if kernel_size != 3:
        raise NotImplementedError(f"Sobel kernel for kernel_size={kernel_size} is not implemented.")

    input_dtype = image.dtype
    compute_dtype = dtypes.float32 if input_dtype == dtypes.uint8 else input_dtype

    if direction == GradientDirection.HORIZONTAL:
        kernel = Tensor(_SOBEL_HORIZONTAL_3x3, dtype=compute_dtype)
    elif direction == GradientDirection.VERTICAL:
        kernel = Tensor(_SOBEL_VERTICAL_3x3, dtype=compute_dtype)
    else:
        raise ValueError(f"Invalid direction: {direction}")

    return apply_convolution_filter(image, kernel, scale, delta, border_mode=PaddingMode.CONSTANT)
