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


def _apply_directional_gradient(
    image: Tensor,
    direction: GradientDirection,
    kernels: tuple[list[list[float | int]], list[list[float | int]]],
    scale: float = 1.0,
    delta: float = 0.0,
    border_mode: PaddingMode = PaddingMode.REFLECT,
) -> Tensor:
    """Convolve with a horizontal or vertical gradient kernel.

    Shared path for operators that only differ by kernel coefficients
    (Sobel, Scharr). Selects the kernel from *direction*, promotes uint8
    inputs to float32 for the kernel dtype, then applies the common
    convolution filter.

    Args:
        image: Input image tensor.
        direction: Gradient direction (HORIZONTAL or VERTICAL).
        kernels: Tuple containing (horizontal_kernel, vertical_kernel) for derivatives.
        scale: Scale factor for output.
        delta: Value added to output.
        border_mode: Padding strategy at image borders.

    Returns:
        Gradient image tensor.

    Raises:
        ValueError: If *direction* is not a known gradient direction.
    """
    input_dtype = image.dtype
    compute_dtype = dtypes.float32 if input_dtype == dtypes.uint8 else input_dtype

    horizontal_kernel, vertical_kernel = kernels

    if direction == GradientDirection.HORIZONTAL:
        kernel = Tensor(horizontal_kernel, dtype=compute_dtype)
    elif direction == GradientDirection.VERTICAL:
        kernel = Tensor(vertical_kernel, dtype=compute_dtype)
    else:
        raise ValueError(f"Invalid direction: {direction}")

    return apply_convolution_filter(image, kernel, scale, delta, border_mode=border_mode)


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
        NotImplementedError: If kernel_size is not 3.
        ValueError: If direction is invalid.
    """
    if kernel_size != 3:
        raise NotImplementedError(f"Sobel kernel for kernel_size={kernel_size} is not implemented.")

    return _apply_directional_gradient(
        image,
        direction,
        kernels=(_SOBEL_HORIZONTAL_3x3, _SOBEL_VERTICAL_3x3),
        scale=scale,
        delta=delta,
        border_mode=PaddingMode.CONSTANT,
    )
