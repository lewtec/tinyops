from tinygrad import Tensor
from tinyops.signal.fft import fft
import math

def ifft(x: Tensor) -> Tensor:
    N = x.shape[-2]

    x_conj = Tensor.stack(x[..., 0], -x[..., 1], dim=-1)

    result_conj = fft(x_conj)

    result = Tensor.stack(result_conj[..., 0], -result_conj[..., 1], dim=-1)

    return result / N
