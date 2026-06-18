from tinygrad import Tensor

NEWTON_SCHULZ_ITERATIONS = 20
NUMERICAL_STABILITY_EPSILON = 1e-12


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
    transpose_permutation = list(range(len(matrix.shape)))
    transpose_permutation[-1], transpose_permutation[-2] = transpose_permutation[-2], transpose_permutation[-1]
    transposed = matrix.permute(transpose_permutation)

    absolute_values = matrix.abs()
    column_norm = absolute_values.sum(axis=-2).max(axis=-1)
    row_norm = absolute_values.sum(axis=-1).max(axis=-1)
    broadcast_shape = list(column_norm.shape) + [1, 1]
    column_norm = column_norm.reshape(broadcast_shape)
    row_norm = row_norm.reshape(broadcast_shape)

    denominator = column_norm * row_norm
    approximation = transposed / (denominator + NUMERICAL_STABILITY_EPSILON)

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
