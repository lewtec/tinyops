"""Shared Newton-Schulz helpers for matrix inverse and pseudo-inverse."""

from tinygrad import Tensor

NEWTON_SCHULZ_ITERATIONS = 20
NUMERICAL_STABILITY_EPSILON = 1e-12


def _transpose_last_two_axes(matrix: Tensor) -> Tensor:
    """Return *matrix* with the last two axes swapped."""
    transpose_permutation = list(range(len(matrix.shape)))
    transpose_permutation[-1], transpose_permutation[-2] = (
        transpose_permutation[-2],
        transpose_permutation[-1],
    )
    return matrix.permute(transpose_permutation)


def _row_and_column_norms(matrix: Tensor) -> tuple[Tensor, Tensor]:
    """Broadcasted max row and column sums for Newton-Schulz scaling."""
    absolute_values = matrix.abs()
    column_norm = absolute_values.sum(axis=-2).max(axis=-1)
    row_norm = absolute_values.sum(axis=-1).max(axis=-1)
    broadcast_shape = list(column_norm.shape) + [1, 1]
    return (
        column_norm.reshape(broadcast_shape),
        row_norm.reshape(broadcast_shape),
    )


def _initial_scaled_transpose_approximation(
    matrix: Tensor,
    *,
    numerical_stability_epsilon: float = 0.0,
) -> Tensor:
    """Scaled transpose used as the Newton-Schulz starting point.

    Scales ``matrix.T`` by the product of max row and column norms so the
    iteration starts near a stable fixed point. When *numerical_stability_epsilon*
    is positive it is added to the denominator (pseudo-inverse path).
    """
    transposed = _transpose_last_two_axes(matrix)
    column_norm, row_norm = _row_and_column_norms(matrix)
    denominator = column_norm * row_norm
    if numerical_stability_epsilon != 0.0:
        denominator = denominator + numerical_stability_epsilon
    return transposed / denominator
