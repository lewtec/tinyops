from tinygrad import Tensor
from tinyops.signal._utils import _to_complex, _complex_mul
import math

def fft(x: Tensor) -> Tensor:
    x = _to_complex(x)
    N = x.shape[-2]

    if N <= 1:
        return x

    if (N & (N - 1)) != 0:
        raise ValueError("FFT size must be a power of 2")

    even = fft(x[..., ::2, :])
    odd = fft(x[..., 1::2, :])

    k = Tensor.arange(N // 2)
    angle = -2 * math.pi * k / N
    twiddle_factors = Tensor.stack(angle.cos(), angle.sin(), dim=-1)

    twiddle_shape = (1,) * (odd.ndim - 2) + (N // 2, 2)
    twiddle_factors = twiddle_factors.reshape(twiddle_shape)

    odd_twiddled = _complex_mul(twiddle_factors, odd)

    return Tensor.cat(even + odd_twiddled, even - odd_twiddled, dim=-2)
