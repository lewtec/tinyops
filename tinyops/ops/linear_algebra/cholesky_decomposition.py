from tinygrad import Tensor


def cholesky_decomposition(matrix: Tensor) -> Tensor:
    """Compute the Cholesky decomposition of a symmetric positive-definite matrix.

    Returns the lower triangular factor L such that ``L @ L.T == matrix``.

    Args:
        matrix: A symmetric positive-definite matrix of shape (n, n).

    Returns:
        The lower triangular Cholesky factor.
    """
    size = matrix.shape[0]
    if size == 0:
        return Tensor.zeros(0, 0, dtype=matrix.dtype)

    columns = []
    for column_index in range(size):
        previous_columns = Tensor.cat(*columns, dim=1) if columns else Tensor.zeros(size, 0, dtype=matrix.dtype)

        squared_sum = (previous_columns[column_index, :] * previous_columns[column_index, :]).sum()
        diagonal_element = (matrix[column_index, column_index] - squared_sum).sqrt()

        if column_index < size - 1:
            cross_product = previous_columns[column_index + 1 :, :] @ previous_columns[column_index, :]
            below_diagonal = (matrix[column_index + 1 :, column_index] - cross_product) / diagonal_element
        else:
            below_diagonal = Tensor.zeros(0, dtype=matrix.dtype)

        zeros_above = Tensor.zeros(column_index, dtype=matrix.dtype)
        diagonal_reshaped = diagonal_element.reshape(1)
        new_column = Tensor.cat(zeros_above, diagonal_reshaped, below_diagonal).realize()
        columns.append(new_column.unsqueeze(1))

    return Tensor.cat(*columns, dim=1)
