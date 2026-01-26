from tinygrad import Tensor


def det(a: Tensor) -> Tensor:
    """
    Calculates the determinant of a square matrix using recursive Laplace expansion.
    Note: This implementation is computationally expensive for large matrices (O(n!))
    and is intended for small matrices typically found in test cases.
    """
    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Input must be a square 2D matrix.")

    n = a.shape[0]
    if n == 0:
        return Tensor(1.0)
    if n == 1:
        return a[0, 0]
    if n == 2:
        return a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]

    total = Tensor(0.0)
    for j in range(n):
        sign = (-1) ** j

        # Create submatrix by excluding the first row and j-th column
        sub_matrix_parts = []
        if j > 0:
            sub_matrix_parts.append(a[1:, :j])
        if j < n - 1:
            sub_matrix_parts.append(a[1:, j + 1 :])

        # a[1:, :j] and a[1:, j+1:] might be empty
        if len(sub_matrix_parts) > 1:
            sub_matrix = sub_matrix_parts[0].cat(sub_matrix_parts[1], dim=1)
        elif len(sub_matrix_parts) == 1:
            sub_matrix = sub_matrix_parts[0]
        else:  # This case for n=1, already handled
            sub_matrix = Tensor([])

        sub_det = det(sub_matrix)
        total += sign * a[0, j] * sub_det

    return total
