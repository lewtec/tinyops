from tinygrad import Tensor

NUMERICAL_STABILITY_EPSILON = 1e-7


def qr_decomposition(matrix: Tensor) -> tuple[Tensor, Tensor]:
    """Compute the reduced QR decomposition using the Gram-Schmidt process.

    Args:
        matrix: Input matrix of shape (m, n).

    Returns:
        A tuple (orthogonal, upper_triangular) where orthogonal has shape
        (m, k) and upper_triangular has shape (k, n), with k = min(m, n).
    """
    rows, columns = matrix.shape
    rank = min(rows, columns)

    if rank == 0:
        return Tensor.zeros(rows, 0, dtype=matrix.dtype), Tensor.zeros(0, columns, dtype=matrix.dtype)

    orthogonal_columns = []
    for column_index in range(rank):
        vector = matrix[:, column_index]
        if column_index > 0:
            previous_orthogonal = Tensor.stack(orthogonal_columns, dim=1)
            projection = previous_orthogonal @ (previous_orthogonal.T @ vector)
            vector = vector - projection

        vector_norm = vector.pow(2).sum().sqrt()
        orthogonal_vector = Tensor.where(
            vector_norm > NUMERICAL_STABILITY_EPSILON,
            vector / vector_norm,
            Tensor.zeros(*vector.shape, dtype=vector.dtype),
        )
        orthogonal_columns.append(orthogonal_vector)

    orthogonal = Tensor.stack(orthogonal_columns, dim=1)
    upper_triangular = orthogonal.T @ matrix

    # Ensure non-negative diagonal elements (convention matching)
    diagonal_values = upper_triangular.flatten()[:: upper_triangular.shape[1] + 1][:rank]
    signs = (diagonal_values < 0).cast(matrix.dtype) * -2 + 1
    orthogonal = orthogonal * signs
    upper_triangular = signs.reshape(-1, 1) * upper_triangular

    return orthogonal, upper_triangular
