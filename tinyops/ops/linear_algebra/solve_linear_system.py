from tinygrad import Tensor

from tinyops.ops.linear_algebra.inverse import inverse


def solve_linear_system(coefficients: Tensor, constants: Tensor) -> Tensor:
    """Solve a linear system of equations ``coefficients @ x = constants``.

    Args:
        coefficients: Coefficient matrix.
        constants: Right-hand side vector or matrix.

    Returns:
        Solution tensor.
    """
    inverse_coefficients = inverse(coefficients)
    if constants.ndim == coefficients.ndim - 1:
        return inverse_coefficients.matmul(constants.unsqueeze(-1)).squeeze(-1)
    return inverse_coefficients.matmul(constants)
