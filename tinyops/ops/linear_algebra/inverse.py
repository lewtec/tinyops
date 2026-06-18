from tinygrad import Tensor

NEWTON_SCHULZ_ITERATIONS = 20


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

    transpose_permutation = list(range(len(matrix.shape)))
    transpose_permutation[-1], transpose_permutation[-2] = transpose_permutation[-2], transpose_permutation[-1]
    transposed = matrix.permute(transpose_permutation)

    absolute_values = matrix.abs()
    column_norm = absolute_values.sum(axis=-2).max(axis=-1)
    row_norm = absolute_values.sum(axis=-1).max(axis=-1)
    broadcast_shape = list(column_norm.shape) + [1, 1]
    column_norm = column_norm.reshape(broadcast_shape)
    row_norm = row_norm.reshape(broadcast_shape)

    approximation = transposed / (column_norm * row_norm)
    identity = Tensor.eye(size)

    for _ in range(NEWTON_SCHULZ_ITERATIONS):
        correction = (2 * identity) - matrix.matmul(approximation)
        approximation = approximation.matmul(correction)

    return approximation
