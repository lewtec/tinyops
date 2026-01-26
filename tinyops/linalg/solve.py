from tinygrad import Tensor

from .inv import inv


def solve(a: Tensor, b: Tensor) -> Tensor:
    """Solve linear system."""
    inv_a = inv(a)
    if b.ndim == a.ndim - 1:
        return inv_a.matmul(b.unsqueeze(-1)).squeeze(-1)
    return inv_a.matmul(b)
