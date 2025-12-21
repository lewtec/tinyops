from tinygrad import Tensor

def filter2d(x: Tensor, kernel: Tensor) -> Tensor:
    """
    Applies a 2D filter to an image, matching cv2.filter2D with default BORDER_REFLECT_101.
    """
    is_grayscale = False
    if x.ndim == 2:
        x = x.unsqueeze(-1)  # H, W -> H, W, 1
        is_grayscale = True

    kH, kW = kernel.shape
    ph, pw = kH // 2, kW // 2

    # Manual padding: BORDER_REFLECT_101
    x_padded = x
    if ph > 0:
        top = x_padded[1:ph+1, :, :].flip(0)
        bottom = x_padded[-ph-1:-1, :, :].flip(0)
        x_padded = Tensor.cat(top, x_padded, bottom, dim=0)

    if pw > 0:
        left = x_padded[:, 1:pw+1, :].flip(1)
        right = x_padded[:, -pw-1:-1, :].flip(1)
        x_padded = Tensor.cat(left, x_padded, right, dim=1)

    # Convolution
    # Reshape for conv2d
    C = x.shape[2]
    x_reshaped = x_padded.permute(2, 0, 1).unsqueeze(0)  # (1, C, H_padded, W_padded)
    kernel_reshaped = kernel.unsqueeze(0).unsqueeze(0).expand(C, 1, kH, kW)

    # Apply convolution without internal padding
    output = x_reshaped.conv2d(kernel_reshaped, groups=C)

    # Reshape back
    output_reshaped = output.squeeze(0).permute(1, 2, 0)

    if is_grayscale:
        output_reshaped = output_reshaped.squeeze(-1)

    return output_reshaped
