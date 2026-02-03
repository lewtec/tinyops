from tinygrad import Tensor

from .inv import inv


def solve(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the exact solution of a system of linear algebraic equations.

    Solves the linear system `Ax = b` for `x`, where `A` is a square matrix and `b`
    is a vector or a matrix. This implementation calculates the inverse of `A` explicitly
    and multiplies it by `b`.

    Args:
        a: Coefficient matrix of shape (..., M, M). Must be square and non-singular.
        b: Ordinate (dependent) variables of shape (..., M) or (..., M, K).

    Returns:
        Solution to the system `Ax = b` with shape matching `b`.

    Warning:
        This function computes `inv(A) @ b`. For large systems or numerically unstable matrices,
        this is less accurate and slower than LU or Cholesky decomposition-based solvers.
        Use only for small-scale problems where an exact inverse is feasible.
    """
    inv_a = inv(a)
    if b.ndim == a.ndim - 1:
        return inv_a.matmul(b.unsqueeze(-1)).squeeze(-1)
    return inv_a.matmul(b)
