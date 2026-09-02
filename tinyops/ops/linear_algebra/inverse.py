from tinygrad import Tensor

from tinyops.ops.linear_algebra._newton_schulz import (
    NEWTON_SCHULZ_ITERATIONS,
    _initial_scaled_transpose_approximation,
)


def inverse(matrix: Tensor) -> Tensor:
    """Compute the multiplicative inverse using Newton-Schulz iteration.

    Args:
        matrix: A square matrix tensor (at least 2D).

    Returns:
        The inverse matrix tensor.

    Raises:
        ValueError: If the input is not at least 2D or not square.
    """
    if len(matrix.shape) < 2:
        raise ValueError("Must be >= 2D")
    size = matrix.shape[-1]
    if matrix.shape[-2] != size:
        raise ValueError("Must be square")

    approximation = _initial_scaled_transpose_approximation(matrix)
    identity = Tensor.eye(size)

    for _ in range(NEWTON_SCHULZ_ITERATIONS):
        correction = (2 * identity) - matrix.matmul(approximation)
        approximation = approximation.matmul(correction)

    return approximation
