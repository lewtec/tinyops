from tinygrad import Tensor, dtypes
from tinyops.image._utils import apply_filter

def get_scharr_kernel(dx: int, dy: int, dtype) -> Tensor:
    if dx == 1 and dy == 0:
        return Tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=dtype)
    if dx == 0 and dy == 1:
        return Tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=dtype)
    raise NotImplementedError(f"Scharr kernel for dx={dx}, dy={dy} is not implemented.")

def scharr(x: Tensor, dx: int, dy: int, scale: float = 1.0, delta: float = 0.0) -> Tensor:
    if not ((dx == 1 and dy == 0) or (dx == 0 and dy == 1)):
        raise ValueError("Scharr filter only supports dx=1, dy=0 or dx=0, dy=1")

    input_dtype = x.dtype
    dtype = dtypes.float32 if input_dtype == dtypes.uint8 else input_dtype

    kernel = get_scharr_kernel(dx, dy, dtype)
    return apply_filter(x, kernel, scale, delta)
