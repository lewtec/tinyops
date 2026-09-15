from tinygrad import Tensor

from tinyops.ops.linear_algebra._newton_schulz import (
    NEWTON_SCHULZ_ITERATIONS,
    NUMERICAL_STABILITY_EPSILON,
    _initial_scaled_transpose_approximation,
)


def pseudo_inverse(matrix: Tensor) -> Tensor:
    """Compute the Moore-Penrose pseudo-inverse using Newton-Schulz iteration.

    Args:
        matrix: Input tensor (at least 2D).

    Returns:
        Pseudo-inverse tensor.

    Raises:
        ValueError: If the input is not at least 2D.
    """
    if len(matrix.shape) < 2:
        raise ValueError("Must be >= 2D")

    rows, columns = matrix.shape[-2], matrix.shape[-1]
    approximation = _initial_scaled_transpose_approximation(
        matrix,
        numerical_stability_epsilon=NUMERICAL_STABILITY_EPSILON,
    )

    if rows >= columns:
        identity = Tensor.eye(columns)
        for _ in range(NEWTON_SCHULZ_ITERATIONS):
            product = approximation.matmul(matrix)
            correction = (2 * identity) - product
            approximation = correction.matmul(approximation)
    else:
        identity = Tensor.eye(rows)
        for _ in range(NEWTON_SCHULZ_ITERATIONS):
            product = matrix.matmul(approximation)
            correction = (2 * identity) - product
            approximation = approximation.matmul(correction)

    return approximation
