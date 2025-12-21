from tinygrad import Tensor, dtypes

def resize(x: Tensor, size: tuple[int, int]) -> Tensor:
    """
    Resizes an image using bilinear interpolation, matching cv2's behavior.
    Args:
        x: Input tensor with shape (H, W, C). Batch dimension is not supported.
        size: A tuple (out_h, out_w) specifying the output size.
    Returns:
        Resized tensor with shape (out_h, out_w, C).
    """
    if len(x.shape) != 3:
        raise ValueError(f"Input tensor must have 3 dimensions (H, W, C), but got {len(x.shape)}")

    in_dtype = x.dtype
    if in_dtype != dtypes.float32:
        x = x.cast(dtypes.float32)

    in_h, in_w, c = x.shape
    out_h, out_w = size

    # Create grids of output coordinates
    grid_y = Tensor.arange(out_h, dtype=dtypes.float32)
    grid_x = Tensor.arange(out_w, dtype=dtypes.float32)

    # Map output coordinates to input coordinates (align_corners=False behavior)
    scale_y = in_h / out_h
    scale_x = in_w / out_w
    in_y = (grid_y + 0.5) * scale_y - 0.5
    in_x = (grid_x + 0.5) * scale_x - 0.5

    # Clip to stay within input bounds
    in_y = in_y.clip(0, in_h - 1)
    in_x = in_x.clip(0, in_w - 1)

    # Get the 4 neighbor coordinates
    y1 = in_y.floor()
    y2 = (y1 + 1).clip(0, in_h - 1)
    x1 = in_x.floor()
    x2 = (x1 + 1).clip(0, in_w - 1)

    # Calculate weights
    dy = (in_y - y1).reshape(out_h, 1, 1)
    dx = (in_x - x1).reshape(1, out_w, 1)

    # Flatten image for gathering
    x_flat = x.reshape(in_h * in_w, c)

    # Cast coordinates to int for indexing
    y1i, y2i = y1.cast(dtypes.int32), y2.cast(dtypes.int32)
    x1i, x2i = x1.cast(dtypes.int32), x2.cast(dtypes.int32)

    # Create grids of indices for gathering
    base_y1 = y1i.reshape(out_h, 1).expand(out_h, out_w)
    base_y2 = y2i.reshape(out_h, 1).expand(out_h, out_w)
    base_x1 = x1i.reshape(1, out_w).expand(out_h, out_w)
    base_x2 = x2i.reshape(1, out_w).expand(out_h, out_w)

    # Calculate flat indices
    idx11 = (base_y1 * in_w + base_x1).flatten()
    idx12 = (base_y1 * in_w + base_x2).flatten()
    idx21 = (base_y2 * in_w + base_x1).flatten()
    idx22 = (base_y2 * in_w + base_x2).flatten()

    # Gather pixel values
    q11 = x_flat[idx11].reshape(out_h, out_w, c)
    q12 = x_flat[idx12].reshape(out_h, out_w, c)
    q21 = x_flat[idx21].reshape(out_h, out_w, c)
    q22 = x_flat[idx22].reshape(out_h, out_w, c)

    # Bilinear interpolation
    r1 = q11 * (1.0 - dx) + q12 * dx
    r2 = q21 * (1.0 - dx) + q22 * dx
    out = r1 * (1.0 - dy) + r2 * dy

    if in_dtype != dtypes.float32:
        if dtypes.is_unsigned(in_dtype):
            out = out.clip(0, 255) # Assuming uint8
        out = out.cast(in_dtype)

    return out
