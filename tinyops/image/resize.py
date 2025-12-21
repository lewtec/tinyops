from tinygrad import Tensor, dtypes

INTER_NEAREST = 0
INTER_LINEAR = 1
INTER_CUBIC = 2
INTER_AREA = 3
INTER_LANCZOS4 = 4

def resize(x: Tensor, dsize: tuple[int, int], interpolation: int = INTER_LINEAR) -> Tensor:
    """
    Resizes an image.

    Args:
        x: Input tensor, shape (H, W) or (H, W, C).
        dsize: Desired output size (out_H, out_W).
        interpolation: Interpolation method. Only INTER_NEAREST and INTER_LINEAR are supported.

    Returns:
        The resized tensor.
    """
    if x.ndim == 2:
        is_hw = True
        x = x.unsqueeze(2)
    else:
        is_hw = False

    H, W, C = x.shape
    out_H, out_W = dsize

    ty, tx = Tensor.meshgrid(Tensor.arange(out_H), Tensor.arange(out_W), indexing='ij')

    if interpolation == INTER_NEAREST:
        scale_y = H / out_H
        scale_x = W / out_W
        sy = ty.cast(dtypes.float32) * scale_y
        sx = tx.cast(dtypes.float32) * scale_x
        iy = sy.floor().cast(dtypes.int32).clip(0, H - 1)
        ix = sx.floor().cast(dtypes.int32).clip(0, W - 1)
        out = x[iy, ix]
    elif interpolation == INTER_LINEAR:
        scale_y = H / out_H
        scale_x = W / out_W
        sy = (ty.cast(dtypes.float32) + 0.5) * scale_y - 0.5
        sx = (tx.cast(dtypes.float32) + 0.5) * scale_x - 0.5
        sy = sy.clip(0, H - 1)
        sx = sx.clip(0, W - 1)

        iy1 = sy.floor().cast(dtypes.int32)
        ix1 = sx.floor().cast(dtypes.int32)

        dy = (sy - iy1).unsqueeze(2)
        dx = (sx - ix1).unsqueeze(2)

        iy2 = (iy1 + 1).clip(0, H - 1)
        ix2 = (ix1 + 1).clip(0, W - 1)

        p11 = x[iy1, ix1]
        p12 = x[iy1, ix2]
        p21 = x[iy2, ix1]
        p22 = x[iy2, ix2]

        w11 = (1 - dy) * (1 - dx)
        w12 = (1 - dy) * dx
        w21 = dy * (1 - dx)
        w22 = dy * dx

        out = p11 * w11 + p12 * w12 + p21 * w21 + p22 * w22
    else:
        raise NotImplementedError(f"Interpolation mode {interpolation} is not supported.")

    if is_hw:
        out = out.squeeze(2)

    return out
