from tinygrad import Tensor, dtypes

def _to_complex(x: Tensor) -> Tensor:
    if x.shape and x.shape[-1] == 2:
        return x.cast(dtypes.float32)
    return Tensor.stack(x, Tensor.zeros(*x.shape, dtype=x.dtype), dim=-1).cast(dtypes.float32)

def _complex_mul(a: Tensor, b: Tensor) -> Tensor:
    a_real, a_imag = a[..., 0], a[..., 1]
    b_real, b_imag = b[..., 0], b[..., 1]
    real = a_real * b_real - a_imag * b_imag
    imag = a_real * b_imag + a_imag * b_real
    return Tensor.stack(real, imag, dim=-1)
