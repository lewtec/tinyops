from tinygrad import Tensor

from tinyops.ops.image._filtering import apply_convolution_filter
from tinyops.ops.image.pad import PaddingMode

# OpenCV getGaussianKernelBitExact fixed tables for sigma <= 0 and n in {1,3,5,7,9}.
_FIXED_GAUSSIAN_WEIGHTS: dict[int, list[float]] = {
    1: [1.0],
    3: [0.25, 0.5, 0.25],
    5: [0.0625, 0.25, 0.375, 0.25, 0.0625],
    7: [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125],
    9: [
        0.015625,
        0.05078125,
        0.1171875,
        0.19921875,
        0.234375,
        0.19921875,
        0.1171875,
        0.05078125,
        0.015625,
    ],
}


def _sigma_from_kernel_size(kernel_length: int) -> float:
    """OpenCV auto-sigma: ((n-1)*0.5 - 1)*0.3 + 0.8 == 0.15*n + 0.35."""
    return 0.15 * kernel_length + 0.35


def _gaussian_1d_weights(kernel_length: int, sigma: float) -> Tensor:
    """Build a normalized 1D Gaussian kernel matching OpenCV getGaussianKernel."""
    if sigma <= 0:
        fixed = _FIXED_GAUSSIAN_WEIGHTS.get(kernel_length)
        if fixed is not None:
            return Tensor(fixed)
        sigma = _sigma_from_kernel_size(kernel_length)

    positions = Tensor.arange(kernel_length) - (kernel_length - 1) / 2
    weights = (-(positions**2) / (2 * sigma**2)).exp()
    return weights / weights.sum()


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
        sigma_x: Standard deviation in the X direction. If <= 0, OpenCV-style
            auto kernels are used (fixed tables for small odd sizes, else
            sigma derived from width).
        sigma_y: Standard deviation in the Y direction. If <= 0, defaults to
            ``sigma_x`` (which may also be auto-derived).

    Returns:
        Blurred image tensor.

    Raises:
        ValueError: If kernel dimensions are not positive and odd.
    """
    # Match OpenCV createGaussianKernels: sigma_y inherits sigma_x when unset.
    if sigma_y <= 0.0:
        sigma_y = sigma_x

    width, height = kernel_size

    if width % 2 == 0 or width <= 0:
        raise ValueError("kernel_size width must be a positive odd number")
    if height % 2 == 0 or height <= 0:
        raise ValueError("kernel_size height must be a positive odd number")

    horizontal_kernel = _gaussian_1d_weights(width, sigma_x).reshape(1, width)
    vertical_kernel = _gaussian_1d_weights(height, sigma_y).reshape(height, 1)

    result = apply_convolution_filter(image, horizontal_kernel, border_mode=PaddingMode.CONSTANT)
    return apply_convolution_filter(result, vertical_kernel, border_mode=PaddingMode.CONSTANT)
