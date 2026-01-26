from tinygrad import Tensor, dtypes

from tinyops.image._utils import _apply_filter_iterative


def get_laplacian_kernel(ksize: int, dtype) -> Tensor:
    if ksize == 1:
        return Tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=dtype)
    raise NotImplementedError(f"Laplacian kernel for ksize={ksize} is not implemented.")


def laplacian(x: Tensor, ksize: int = 1, scale: float = 1.0, delta: float = 0.0) -> Tensor:
    if ksize != 1:
        raise ValueError("ksize must be 1")

    input_dtype = x.dtype
    dtype = dtypes.float32 if input_dtype == dtypes.uint8 else input_dtype

    kernel = get_laplacian_kernel(ksize, dtype)
    return _apply_filter_iterative(x, kernel, scale, delta)
