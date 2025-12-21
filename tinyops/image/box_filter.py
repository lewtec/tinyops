from tinygrad import Tensor
from tinyops.image._padding import pad_reflect_101

def box_filter(x: Tensor, ksize: tuple[int, int], normalize: bool = True) -> Tensor:
    """
    Blurs an image using the box filter.

    Args:
        x: Input image tensor in (B, C, H, W) format.
        ksize: Blurring kernel size.
        normalize: If true, the kernel is normalized by its area.

    Returns:
        Blurred image tensor.
    """
    channels = x.shape[1]
    kernel = Tensor.ones(channels, 1, *ksize)
    if normalize:
        kernel /= (ksize[0] * ksize[1])

    # Add padding to match cv2.boxFilter's default behavior.
    padding = (ksize[1] // 2, ksize[1] // 2, ksize[0] // 2, ksize[0] // 2)
    x = pad_reflect_101(x, padding)

    return x.conv2d(kernel, groups=channels)
