from tinygrad import Tensor
from tinyops.image._padding import pad_reflect_101

def gaussian_blur(x: Tensor, ksize: tuple[int, int], sigmaX: float, sigmaY: float | None = None) -> Tensor:
    """
    Blurs an image using a Gaussian filter.

    Args:
        x: Input image tensor in (B, C, H, W) format.
        ksize: Gaussian kernel size.
        sigmaX: Gaussian kernel standard deviation in X direction.
        sigmaY: Gaussian kernel standard deviation in Y direction. If None, sigmaY is set to sigmaX.

    Returns:
        Blurred image tensor.
    """
    if sigmaY is None:
        sigmaY = sigmaX

    channels = x.shape[1]

    # Create 1D Gaussian kernels
    kernel_x = Tensor.arange(ksize[0]).sub(ksize[0] // 2)
    kernel_x = kernel_x.pow(2).div(-2 * sigmaX**2).exp()
    kernel_x = kernel_x / kernel_x.sum()

    kernel_y = Tensor.arange(ksize[1]).sub(ksize[1] // 2)
    kernel_y = kernel_y.pow(2).div(-2 * sigmaY**2).exp()
    kernel_y = kernel_y / kernel_y.sum()

    # Create 2D Gaussian kernel
    kernel = kernel_x.unsqueeze(1) * kernel_y.unsqueeze(0)
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat((channels, 1, 1, 1))

    # Add padding to match cv2.GaussianBlur's default behavior.
    padding = (ksize[1] // 2, ksize[1] // 2, ksize[0] // 2, ksize[0] // 2)
    x = pad_reflect_101(x, padding)

    return x.conv2d(kernel, groups=channels)
