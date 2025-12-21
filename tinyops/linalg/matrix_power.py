from tinygrad import Tensor
from .inv import inv

def matrix_power(a: Tensor, n: int) -> Tensor:
    """
    Raise a square matrix to the (integer) power n.
    """
    if len(a.shape) < 2:
        raise ValueError("a must have at least 2 dimensions")
    if a.shape[-2] != a.shape[-1]:
        raise ValueError("Last 2 dimensions of the array must be square")

    M = a.shape[-1]

    if n == 0:
        I = Tensor.eye(M, dtype=a.dtype, device=a.device)
        # Broadcast to match batch dimensions
        batch_shape = a.shape[:-2]
        if batch_shape:
            I = I.reshape((1,) * len(batch_shape) + (M, M))
            I = I.expand(a.shape)
        return I

    if n < 0:
        a = inv(a)
        n = abs(n)

    res = None
    current_pow = a
    while n > 0:
        if n % 2 == 1:
            if res is None:
                res = current_pow
            else:
                res = res.matmul(current_pow)

        n //= 2
        if n > 0:
            current_pow = current_pow.matmul(current_pow)

    return res
