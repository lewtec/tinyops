from tinygrad import Tensor


def box_filter(x: Tensor, ksize: tuple[int, int]) -> Tensor:
    """
    Blurs an image using the box filter. This implementation uses zero-padding,
    which is equivalent to cv2.BORDER_CONSTANT with value 0.

    Args:
      x: Input image tensor (H, W, C) or (H, W).
      ksize: Blurring kernel size.

    Returns:
      Blurred image tensor.
    """
    input_shape_len = len(x.shape)
    if input_shape_len == 2:
        x = x.unsqueeze(2)  # Add channel dimension

    in_channels = x.shape[2]
    h, w = ksize

    # Create the box kernel
    kernel_2d = Tensor.ones(h, w, requires_grad=False) / (h * w)
    # Shape for depthwise conv: (out_channels, 1, H, W) where out_channels == in_channels
    kernel = kernel_2d.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)

    # Reshape input tensor for convolution from (H, W, C) to (1, C, H, W)
    x_reshaped = x.permute(2, 0, 1).unsqueeze(0)

    # Calculate padding for 'SAME' output size.
    # OpenCV's anchor for even kernels is at k/2 - 1, which means it pulls pixels more
    # from the bottom/right. This corresponds to larger padding at the top/left.
    pad_top = h // 2
    pad_bottom = (h - 1) // 2
    pad_left = w // 2
    pad_right = (w - 1) // 2
    padding = (pad_left, pad_right, pad_top, pad_bottom)

    # Perform depthwise convolution
    result = x_reshaped.conv2d(kernel, padding=padding, groups=in_channels)

    # Reshape back to (H, W, C)
    result = result.squeeze(0).permute(1, 2, 0)

    if input_shape_len == 2:
        result = result.squeeze(2)  # Remove channel dimension

    return result
