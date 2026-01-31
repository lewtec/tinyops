from tinygrad import Tensor, dtypes

from tinyops.image.pad import pad


def apply_filter(
    x: Tensor, kernel: Tensor, scale: float = 1.0, delta: float = 0.0, padding_mode: str = "reflect", padding=None
) -> Tensor:
    """
    Applies a 2D filter to the image.

    Args:
        x: Input tensor (H, W), (H, W, C), or (N, H, W, C).
        kernel: Filter kernel (H, W).
        scale: Scale factor.
        delta: Delta value.
        padding_mode: Padding mode ('reflect', 'constant').
        padding: Tuple (left, top, right, bottom) or None. If None, calculated for SAME padding.
    """
    input_dtype = x.dtype
    if input_dtype == dtypes.uint8:
        x = x.cast(dtypes.float32)

    kH, kW = kernel.shape

    # Default padding (Same)
    if padding is None:
        ph_top = kH // 2
        ph_bottom = (kH - 1) // 2
        pw_left = kW // 2
        pw_right = (kW - 1) // 2
        padding = (pw_left, ph_top, pw_right, ph_bottom)

    # Prepare for convolution
    # If mode is constant (0), we let conv2d handle it for efficiency.
    # Otherwise we manual pad.

    conv_padding = (0, 0, 0, 0)
    padded_x = x

    # Note: We assume 'constant' padding implies value 0, which matches conv2d default.
    if padding_mode == "constant":
        # Convert (l, t, r, b) to conv2d format (l, r, t, b)
        conv_padding = (padding[0], padding[2], padding[1], padding[3])
    else:
        # Manual pad
        # Handle batch dimension for pad() which expects (H, W, ...)
        if x.ndim == 4:  # (N, H, W, C)
            # Permute to (H, W, N, C) so pad() sees H, W as dims 0, 1
            x_perm = x.permute(1, 2, 0, 3)
            padded_x_perm = pad(x_perm, padding, padding_mode=padding_mode)
            padded_x = padded_x_perm.permute(2, 0, 1, 3)
        else:
            padded_x = pad(x, padding, padding_mode=padding_mode)

    # Prepare Input for conv2d (N, C, H, W)
    if padded_x.ndim == 2:  # H, W
        # (1, 1, H, W)
        x_in = padded_x.reshape(1, 1, *padded_x.shape)
        groups = 1
        orig_perm = None
    elif padded_x.ndim == 3:  # H, W, C
        # (1, C, H, W)
        x_in = padded_x.permute(2, 0, 1).unsqueeze(0)
        groups = padded_x.shape[2]
        orig_perm = (1, 2, 0)
    elif padded_x.ndim == 4:  # N, H, W, C
        # (N, C, H, W)
        x_in = padded_x.permute(0, 3, 1, 2)
        groups = padded_x.shape[3]
        orig_perm = (0, 2, 3, 1)
    else:
        raise ValueError(f"Unsupported input shape: {x.shape}")

    # Prepare Kernel
    # (groups, 1, kH, kW) for depthwise
    k = kernel.expand(groups, 1, kH, kW)

    # Convolution
    out = x_in.conv2d(k, padding=conv_padding, groups=groups)

    # Restore Shape
    if orig_perm:
        if len(orig_perm) == 3:  # HWC
            out = out.squeeze(0).permute(*orig_perm)
        else:  # NHWC
            out = out.permute(*orig_perm)
    else:  # HW
        out = out.reshape(out.shape[2], out.shape[3])

    return out * scale + delta


# Alias for backward compatibility if needed, though we will update callers
_apply_filter_iterative = apply_filter
