from tinygrad import Tensor

from tinyops.image._utils import apply_filter


def gaussian_blur(x: Tensor, ksize: tuple[int, int], sigmaX: float, sigmaY: float = 0.0) -> Tensor:
    """
    Blurs an image using a Gaussian filter.

    Args:
        x (Tensor): Input image tensor (H, W, C) or (H, W).
        ksize (tuple[int, int]): Gaussian kernel size.
        sigmaX (float): Gaussian kernel standard deviation in X direction.
        sigmaY (float): Gaussian kernel standard deviation in Y direction. If zero,
                        it is set to be the same as sigmaX.

    Returns:
        Tensor: Blurred image tensor.
    """
    if sigmaY == 0.0:
        sigmaY = sigmaX

    kx = ksize[0]
    ky = ksize[1]

    if kx % 2 == 0 or kx <= 0:
        raise ValueError("ksize width must be a positive odd number")
    if ky % 2 == 0 or ky <= 0:
        raise ValueError("ksize height must be a positive odd number")

    # Create horizontal kernel (1, kx)
    ax = Tensor.arange(kx) - (kx - 1) / 2
    g_x = (-(ax**2) / (2 * sigmaX**2)).exp()
    g_x = g_x / g_x.sum()
    g_x = g_x.reshape(1, kx)

    # Create vertical kernel (ky, 1)
    ay = Tensor.arange(ky) - (ky - 1) / 2
    g_y = (-(ay**2) / (2 * sigmaY**2)).exp()
    g_y = g_y / g_y.sum()
    g_y = g_y.reshape(ky, 1)

    # Apply separable convolution
    # First apply horizontal blur
    out = apply_filter(x, g_x, padding_mode="constant")
    # Then apply vertical blur
    out = apply_filter(out, g_y, padding_mode="constant")

    return out
