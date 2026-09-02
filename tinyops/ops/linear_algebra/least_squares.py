from tinygrad import Tensor

from tinyops.ops.linear_algebra.pseudo_inverse import pseudo_inverse


def least_squares(coefficients: Tensor, observations: Tensor) -> Tensor:
    """Compute the least-squares solution to a linear system.

    Args:
        coefficients: Coefficient matrix.
        observations: Right-hand side vector or matrix.

    Returns:
        Least-squares solution tensor.
    """
    pinv = pseudo_inverse(coefficients)
    if observations.ndim == coefficients.ndim - 1:
        return pinv.matmul(observations.unsqueeze(-1)).squeeze(-1)
    return pinv.matmul(observations)
