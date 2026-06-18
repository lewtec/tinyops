from tinygrad import Tensor


def determinant(matrix: Tensor) -> Tensor:
    """Compute the determinant of a square matrix via Laplace expansion.

    This is O(n!) and intended only for small matrices.

    Args:
        matrix: A 2D square tensor.

    Returns:
        Scalar tensor containing the determinant.

    Raises:
        ValueError: If the input is not a square 2D matrix.
    """
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be a square 2D matrix.")

    size = matrix.shape[0]
    if size == 0:
        return Tensor(1.0)
    if size == 1:
        return matrix[0, 0]
    if size == 2:
        return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]

    total = Tensor(0.0)
    for column in range(size):
        sign = (-1) ** column
        sub_matrix_parts = []
        if column > 0:
            sub_matrix_parts.append(matrix[1:, :column])
        if column < size - 1:
            sub_matrix_parts.append(matrix[1:, column + 1:])

        if len(sub_matrix_parts) > 1:
            sub_matrix = sub_matrix_parts[0].cat(sub_matrix_parts[1], dim=1)
        elif len(sub_matrix_parts) == 1:
            sub_matrix = sub_matrix_parts[0]
        else:
            sub_matrix = Tensor([])

        total += sign * matrix[0, column] * determinant(sub_matrix)

    return total
