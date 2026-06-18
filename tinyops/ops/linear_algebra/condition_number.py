from tinygrad import Tensor

from tinyops.ops.linear_algebra.inverse import inverse
from tinyops.ops.linear_algebra.norm import norm


def condition_number(matrix: Tensor, order: int | float | str | None = None) -> Tensor:
    """Compute the condition number of a matrix.

    The condition number is ``norm(matrix) * norm(inverse(matrix))``.

    Args:
        matrix: Input matrix tensor.
        order: Order of the norm (same as :func:`norm`). The 2-norm
            is not supported.

    Returns:
        Condition number scalar tensor.

    Raises:
        NotImplementedError: For order=None or order=2 (requires SVD).
    """
    if order is None:
        raise NotImplementedError("Default condition number (2-norm) not supported. Use order='fro', 1, -1, inf, -inf.")
    if order == 2:
        raise NotImplementedError("2-norm condition number not supported.")

    inverse_matrix = inverse(matrix)
    return norm(matrix, order=order) * norm(inverse_matrix, order=order)
