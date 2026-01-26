from tinygrad import Tensor

from .inv import inv
from .norm import norm


def cond(x: Tensor, p=None) -> Tensor:
    """
    Compute the condition number of a matrix.
    """
    if p is None:
        # p=None -> 2-norm. SVD needed.
        raise NotImplementedError("Default condition number (2-norm) not supported. Use p='fro', 1, -1, inf, -inf.")

    if p == 2:
        raise NotImplementedError("2-norm condition number not supported.")

    # cond = norm(x) * norm(inv(x))
    inv_x = inv(x)
    return norm(x, ord=p) * norm(inv_x, ord=p)
