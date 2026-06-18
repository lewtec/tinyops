from tinygrad import Tensor

from tinyops.ops.image._filtering import apply_convolution_filter
from tinyops.ops.image.pad import PaddingMode


def spatial_filter(image: Tensor, kernel: Tensor) -> Tensor:
    """Apply an arbitrary 2D spatial filter to an image.

    Uses reflect padding by default, matching common image processing
    conventions.

    Args:
        image: Input image tensor (H, W), (H, W, C), or (N, H, W, C).
        kernel: 2D filter kernel tensor.

    Returns:
        Filtered image with the same shape as the input.
    """
    return apply_convolution_filter(image, kernel, border_mode=PaddingMode.REFLECT)
