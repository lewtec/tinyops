from tinygrad import Tensor

from tinyops.ops.linear_algebra.qr_decomposition import qr_decomposition


def matrix_rank(matrix: Tensor, tolerance: float = 1e-5) -> Tensor:
    """Compute the rank of a matrix via QR decomposition.

    Args:
        matrix: Input 2D tensor.
        tolerance: Threshold below which diagonal values of R are
            considered zero.

    Returns:
        Scalar tensor containing the rank.
    """
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return Tensor(0)

    _, upper_triangular = qr_decomposition(matrix)
    diagonal_elements = upper_triangular.flatten()[:: upper_triangular.shape[1] + 1]
    return (diagonal_elements.abs() > tolerance).sum()
