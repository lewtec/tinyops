from tinygrad import Tensor
from tinyops.image.box_filter import box_filter

def blur(x: Tensor, ksize: tuple[int, int]) -> Tensor:
    """
    Blurs an image using the normalized box filter.

    Args:
        x: Input image tensor.
        ksize: Blurring kernel size.

    Returns:
        Blurred image tensor.
    """
    return box_filter(x, ksize, normalize=True)
