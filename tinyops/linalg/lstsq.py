from tinygrad import Tensor

from .pinv import pinv


def lstsq(a: Tensor, b: Tensor) -> Tensor:
    """Least-squares solution."""
    inv_a = pinv(a)
    if b.ndim == a.ndim - 1:
        return inv_a.matmul(b.unsqueeze(-1)).squeeze(-1)
    return inv_a.matmul(b)
