from tinygrad import Tensor

def pad_reflect_101(x: Tensor, padding: tuple[int, int, int, int]) -> Tensor:
    """
    Pads a tensor using the 'reflect_101' method, which corresponds to
    OpenCV's BORDER_REFLECT_101. This mode does not repeat the border
    pixels in the reflection.

    Args:
        x: Input tensor in (B, C, H, W) format.
        padding: A tuple of 4 integers (pad_left, pad_right, pad_top, pad_bottom).

    Returns:
        Padded tensor.
    """
    p_l, p_r, p_t, p_b = padding

    # Horizontal padding
    left_pad = x[:, :, :, 1:p_l+1].flip(3)
    right_pad = x[:, :, :, -(p_r+1):-1].flip(3)
    x = Tensor.cat(left_pad, x, right_pad, dim=3)

    # Vertical padding
    top_pad = x[:, :, 1:p_t+1, :].flip(2)
    bottom_pad = x[:, :, -(p_b+1):-1, :].flip(2)
    x = Tensor.cat(top_pad, x, bottom_pad, dim=2)

    return x
