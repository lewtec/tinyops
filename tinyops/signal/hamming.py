import math

from tinygrad import Tensor


def hamming(M: int, sym: bool = True) -> Tensor:
    if M < 1:
        return Tensor([])
    if M == 1:
        return Tensor.ones(1)

    denominator = M - 1 if sym else M
    n = Tensor.arange(M)

    w = 0.54 - 0.46 * (2 * math.pi * n / denominator).cos()

    return w
