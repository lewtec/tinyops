from tinygrad import Tensor

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

    # Create 1D Gaussian kernels
    kx = ksize[0]
    ky = ksize[1]

    if kx % 2 == 0 or kx <= 0: raise ValueError("ksize width must be a positive odd number")
    if ky % 2 == 0 or ky <= 0: raise ValueError("ksize height must be a positive odd number")

    # Create horizontal kernel
    ax = Tensor.arange(kx) - (kx - 1) / 2
    g_x = (-ax**2 / (2 * sigmaX**2)).exp()
    g_x = g_x / g_x.sum()

    # Create vertical kernel
    ay = Tensor.arange(ky) - (ky - 1) / 2
    g_y = (-ay**2 / (2 * sigmaY**2)).exp()
    g_y = g_y / g_y.sum()

    # Apply separable convolution
    # Apply horizontal blur
    # Reshape input tensor for convolution from (H, W, C) to (1, C, H, W)
    input_shape_len = len(x.shape)
    if input_shape_len == 2:
        x = x.unsqueeze(2) # Add channel dimension

    in_channels = x.shape[2]
    x_reshaped = x.permute(2, 0, 1).unsqueeze(0)

    # kernel shape for depthwise conv: (in_channels, 1, H, W)
    kernel_x = g_x.reshape(1, 1, 1, kx).repeat(in_channels, 1, 1, 1)
    padding_x = ((kx - 1) // 2, kx // 2, 0, 0)

    intermediate = x_reshaped.conv2d(kernel_x, padding=padding_x, groups=in_channels)

    # Apply vertical blur
    kernel_y = g_y.reshape(1, 1, ky, 1).repeat(in_channels, 1, 1, 1)
    padding_y = (0, 0, (ky - 1) // 2, ky // 2)

    result = intermediate.conv2d(kernel_y, padding=padding_y, groups=in_channels)

    # Reshape back to (H, W, C)
    result = result.squeeze(0).permute(1, 2, 0)

    if input_shape_len == 2:
        result = result.squeeze(2)

    return result
