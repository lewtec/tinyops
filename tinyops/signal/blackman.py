import numpy as np
from tinygrad import Tensor


def blackman(M: int, sym: bool = True) -> Tensor:
    if M < 1:
        return Tensor([])
    if M == 1:
        return Tensor.ones(1)

    denominator = M - 1 if sym else M
    n = Tensor.arange(M)

    w = 0.42 - 0.5 * (2 * np.pi * n / denominator).cos() + 0.08 * (4 * np.pi * n / denominator).cos()

    return w
