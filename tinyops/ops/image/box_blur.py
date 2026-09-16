from tinygrad import Tensor

from tinyops.ops.image._filtering import apply_convolution_filter
from tinyops.ops.image.pad import PaddingMode


def box_blur(
    image: Tensor,
    kernel_size: tuple[int, int],
    *,
    normalize: bool = True,
) -> Tensor:
    """Blur an image using a box filter.

    Args:
        image: Input image tensor (H, W) or (H, W, C).
        kernel_size: Filter size as (height, width).
        normalize: When True (default), average over the kernel window
            (OpenCV ``blur`` / ``boxFilter(..., normalize=True)``). When
            False, sum over the window (``boxFilter(..., normalize=False)``).

    Returns:
        Filtered image tensor.
    """
    height, width = kernel_size
    kernel = Tensor.ones(height, width, requires_grad=False)
    if normalize:
        kernel = kernel / (height * width)
    return apply_convolution_filter(image, kernel, border_mode=PaddingMode.CONSTANT)
