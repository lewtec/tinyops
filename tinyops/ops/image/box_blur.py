from tinygrad import Tensor

from tinyops.ops.image._filtering import apply_convolution_filter
from tinyops.ops.image.pad import PaddingMode


def box_blur(image: Tensor, kernel_size: tuple[int, int]) -> Tensor:
    """Blur an image using a box (average) filter.

    Args:
        image: Input image tensor (H, W) or (H, W, C).
        kernel_size: Filter size as (height, width).

    Returns:
        Blurred image tensor.
    """
    height, width = kernel_size
    kernel = Tensor.ones(height, width, requires_grad=False) / (height * width)
    return apply_convolution_filter(image, kernel, border_mode=PaddingMode.CONSTANT)
