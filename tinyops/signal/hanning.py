from tinygrad import Tensor
import math

def hanning(M: int, sym: bool = True) -> Tensor:
    if M < 1:
        return Tensor([])
    if M == 1:
        return Tensor.ones(1)

    denominator = M - 1 if sym else M
    n = Tensor.arange(M)

    w = 0.5 - 0.5 * (2 * math.pi * n / denominator).cos()

    return w
