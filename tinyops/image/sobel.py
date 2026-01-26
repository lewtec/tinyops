from tinygrad import Tensor, dtypes

from tinyops.image._utils import _apply_filter_iterative


def get_sobel_kernel(dx: int, dy: int, ksize: int, dtype) -> Tensor:
    if ksize % 2 == 0 or ksize <= 1:
        raise ValueError("ksize must be odd and > 1")

    if ksize == 3:
        if dx == 1 and dy == 0:
            return Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype)
        if dx == 0 and dy == 1:
            return Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype)

    raise NotImplementedError(f"Sobel kernel for ksize={ksize} is not implemented.")


def sobel(x: Tensor, dx: int, dy: int, ksize: int = 3, scale: float = 1.0, delta: float = 0.0) -> Tensor:
    if not ((dx == 1 and dy == 0) or (dx == 0 and dy == 1)):
        raise ValueError("Sobel filter only supports dx=1, dy=0 or dx=0, dy=1")

    input_dtype = x.dtype
    dtype = dtypes.float32 if input_dtype == dtypes.uint8 else input_dtype

    kernel = get_sobel_kernel(dx, dy, ksize, dtype)
    return _apply_filter_iterative(x, kernel, scale, delta, padding_mode="constant")
