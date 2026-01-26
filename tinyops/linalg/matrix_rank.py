from tinygrad import Tensor

from tinyops.linalg.qr import qr


def matrix_rank(a: Tensor, tol: float = 1e-5) -> Tensor:
    """
    Computes the matrix rank of a matrix.

    Args:
        a: A matrix (m, n).
        tol: The tolerance below which singular values are considered zero.

    Returns:
        The rank of the matrix.
    """
    if a.shape[0] == 0 or a.shape[1] == 0:
        return Tensor(0)

    _, r = qr(a)
    diag = r.flatten()[:: r.shape[1] + 1]
    rank = (diag.abs() > tol).sum()
    return rank
