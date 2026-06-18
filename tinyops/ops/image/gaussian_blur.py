from tinygrad import Tensor

from tinyops.ops.image._filtering import apply_convolution_filter
from tinyops.ops.image.pad import PaddingMode


def gaussian_blur(
    image: Tensor,
    kernel_size: tuple[int, int],
    sigma_x: float,
    sigma_y: float = 0.0,
) -> Tensor:
    """Blur an image using a separable Gaussian filter.

    Applies horizontal then vertical 1D Gaussian kernels for efficiency.

    Args:
        image: Input image tensor.
        kernel_size: Kernel size as (width, height). Both must be positive odd.
        sigma_x: Standard deviation in the X direction.
        sigma_y: Standard deviation in the Y direction.
            If 0, defaults to sigma_x.

    Returns:
        Blurred image tensor.

    Raises:
        ValueError: If kernel dimensions are not positive and odd.
    """
    if sigma_y == 0.0:
        sigma_y = sigma_x

    width, height = kernel_size

    if width % 2 == 0 or width <= 0:
        raise ValueError("kernel_size width must be a positive odd number")
    if height % 2 == 0 or height <= 0:
        raise ValueError("kernel_size height must be a positive odd number")

    # Horizontal kernel (1, width)
    horizontal_positions = Tensor.arange(width) - (width - 1) / 2
    horizontal_weights = (-(horizontal_positions ** 2) / (2 * sigma_x ** 2)).exp()
    horizontal_weights = horizontal_weights / horizontal_weights.sum()
    horizontal_kernel = horizontal_weights.reshape(1, width)

    # Vertical kernel (height, 1)
    vertical_positions = Tensor.arange(height) - (height - 1) / 2
    vertical_weights = (-(vertical_positions ** 2) / (2 * sigma_y ** 2)).exp()
    vertical_weights = vertical_weights / vertical_weights.sum()
    vertical_kernel = vertical_weights.reshape(height, 1)

    result = apply_convolution_filter(image, horizontal_kernel, border_mode=PaddingMode.CONSTANT)
    return apply_convolution_filter(result, vertical_kernel, border_mode=PaddingMode.CONSTANT)
