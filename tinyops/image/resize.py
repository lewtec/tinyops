from enum import Enum
from functools import partial

from tinygrad import Tensor, dtypes


def resize_nearest(x, out_H, out_W, H, W, ty, tx) -> Tensor:
    """
    Nearest-neighbor interpolation kernel.

    Maps output coordinates (ty, tx) to input coordinates using the nearest integer index.
    """
    scale_y = H / out_H
    scale_x = W / out_W
    sy = ty.cast(dtypes.float32) * scale_y
    sx = tx.cast(dtypes.float32) * scale_x
    iy = sy.floor().cast(dtypes.int32).clip(0, H - 1)
    ix = sx.floor().cast(dtypes.int32).clip(0, W - 1)
    return x[iy, ix]


def resize_linear(x, out_H, out_W, H, W, ty, tx) -> Tensor:
    """
    Bilinear interpolation kernel.

    Maps output coordinates (ty, tx) to input coordinates and computes the weighted average
    of the 2x2 neighborhood.
    """
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

    return p11 * w11 + p12 * w12 + p21 * w21 + p22 * w22


def resize_not_implemented(*args, **kwargs):
    raise NotImplementedError("This interpolation mode is not supported.")


class Interpolation(Enum):
    NEAREST = partial(resize_nearest)
    LINEAR = partial(resize_linear)
    CUBIC = partial(resize_not_implemented)
    AREA = partial(resize_not_implemented)
    LANCZOS4 = partial(resize_not_implemented)

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


# Backward compatibility constants
INTER_NEAREST = 0
INTER_LINEAR = 1
INTER_CUBIC = 2
INTER_AREA = 3
INTER_LANCZOS4 = 4

_INT_TO_INTERPOLATION = {
    INTER_NEAREST: Interpolation.NEAREST,
    INTER_LINEAR: Interpolation.LINEAR,
    INTER_CUBIC: Interpolation.CUBIC,
    INTER_AREA: Interpolation.AREA,
    INTER_LANCZOS4: Interpolation.LANCZOS4
}

def resize(x: Tensor, dsize: tuple[int, int], interpolation: int | Interpolation = INTER_LINEAR) -> Tensor:
    """
    Resizes an image.

    Args:
        x: Input tensor, shape (H, W) or (H, W, C).
        dsize: Desired output size in (height, width) format.
               **Note**: This differs from OpenCV's `(width, height)` convention.
        interpolation: Interpolation method. Only INTER_NEAREST and INTER_LINEAR are supported.

    Returns:
        The resized tensor.

    Raises:
        NotImplementedError: If the interpolation method is not supported.
    """
    if x.ndim == 2:
        is_hw = True
        x = x.unsqueeze(2)
    else:
        is_hw = False

    H, W, C = x.shape
    out_H, out_W = dsize

    ty, tx = Tensor.meshgrid(Tensor.arange(out_H), Tensor.arange(out_W), indexing="ij")

    if isinstance(interpolation, int):
        if interpolation in _INT_TO_INTERPOLATION:
            interp_enum = _INT_TO_INTERPOLATION[interpolation]
        else:
            raise NotImplementedError(f"Interpolation mode {interpolation} is not supported.")
    elif isinstance(interpolation, Interpolation):
        interp_enum = interpolation
    else:
        raise TypeError(f"Invalid type for interpolation: {type(interpolation)}")

    out = interp_enum(x, out_H, out_W, H, W, ty, tx)

    if is_hw:
        out = out.squeeze(2)

    return out
