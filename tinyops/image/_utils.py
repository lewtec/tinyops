from tinygrad import Tensor, dtypes

def apply_filter(x: Tensor, kernel: Tensor, scale: float = 1.0, delta: float = 0.0, padding_mode: str = 'reflect', padding=None) -> Tensor:
    input_dtype = x.dtype
    if input_dtype == dtypes.uint8:
        x = x.cast(dtypes.float32)

    kH, kW = kernel.shape

    # Determine Padding (L, R, T, B)
    if padding is None:
        ph, pw = kH // 2, kW // 2
        padding_val = (pw, pw, ph, ph)
    else:
        # Normalize padding to (L, R, T, B)
        if isinstance(padding, int):
            padding_val = (padding, padding, padding, padding)
        elif len(padding) == 2:
            padding_val = (padding[0], padding[0], padding[1], padding[1])
        elif len(padding) == 4:
            padding_val = padding
        else:
             raise ValueError("Padding must be int, tuple(2), or tuple(4)")

    # Reshape logic to NCHW
    orig_shape = x.shape
    ndim = len(orig_shape)

    if ndim == 2: # (H, W)
        H, W = orig_shape
        C = 1
        x = x.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
        groups = 1
    elif ndim == 3: # (H, W, C)
        H, W, C = orig_shape
        x = x.permute(2, 0, 1).unsqueeze(0) # (1, C, H, W)
        groups = C
    elif ndim == 4: # (N, H, W, C)
        N, H, W, C = orig_shape
        x = x.permute(0, 3, 1, 2) # (N, C, H, W)
        groups = C
    else:
        raise ValueError(f"Unsupported input shape: {orig_shape}")

    # Prepare kernel
    # Kernel is (kH, kW). For grouped conv: (C, 1, kH, kW)
    kernel_reshaped = kernel.reshape(1, 1, kH, kW).repeat(groups, 1, 1, 1)

    # Convolve
    if padding_mode == 'constant':
         # Use efficient conv2d padding
         # padding_val is (L, R, T, B)
         # conv2d padding arg expects (L, R, T, B) usually in tinygrad/onnx?
         # Tinygrad conv2d signature: padding.
         # If tuple(4), it is ((top, bottom), (left, right)) or (L, R, T, B)?
         # Checking box_filter.py: padding = (pad_left, pad_right, pad_top, pad_bottom).
         # It calls conv2d(..., padding=padding).
         # So we pass (L, R, T, B).
         y = x.conv2d(kernel_reshaped, padding=padding_val, groups=groups)

    elif padding_mode == 'reflect':
         # Manual reflect padding on NCHW
         L, R, T, B = padding_val

         # Pad W (dim 3)
         if L > 0:
            x = x[..., 1:L+1].flip(-1).cat(x, dim=-1)
         if R > 0:
            x = x.cat(x[..., -R-1:-1].flip(-1), dim=-1)

         # Pad H (dim 2)
         if T > 0:
            x = x[..., 1:T+1, :].flip(-2).cat(x, dim=-2)
         if B > 0:
            x = x.cat(x[..., -B-1:-1, :].flip(-2), dim=-2)

         y = x.conv2d(kernel_reshaped, groups=groups)

    else:
         raise ValueError(f"Unsupported padding mode: {padding_mode}")

    # Reshape back to original layout
    if ndim == 2:
        y = y.squeeze(0).squeeze(0) # (H, W)
    elif ndim == 3:
        y = y.squeeze(0).permute(1, 2, 0) # (H, W, C)
    elif ndim == 4:
        y = y.permute(0, 2, 3, 1) # (N, H, W, C)

    return y * scale + delta

# Alias for backward compatibility if needed, but we will refactor all callers.
_apply_filter_iterative = apply_filter
