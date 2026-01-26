from tinygrad import Tensor


def qr(a: Tensor) -> tuple[Tensor, Tensor]:
    """
    Computes the QR decomposition of a matrix using the Gram-Schmidt process.

    Args:
        a: A matrix (m, n).

    Returns:
        A tuple (Q, R) where Q is an orthogonal matrix (m, k) and R is an
        upper triangular matrix (k, n), with k = min(m, n).
        This is the 'reduced' QR decomposition.
    """
    m, n = a.shape
    k = min(m, n)

    if k == 0:
        return Tensor.zeros(m, 0, dtype=a.dtype), Tensor.zeros(0, n, dtype=a.dtype)

    q_cols = []
    for j in range(k):
        v = a[:, j]
        if j > 0:
            q_prev = Tensor.stack(q_cols, dim=1)
            proj = q_prev @ (q_prev.T @ v)
            v = v - proj

        norm_v = v.pow(2).sum().sqrt()
        # If the norm is close to zero, the vector is linearly dependent.
        # Treat the corresponding column of Q as a zero vector to reflect this.
        q_j = Tensor.where(norm_v > 1e-7, v / norm_v, Tensor.zeros(*v.shape, dtype=v.dtype))
        q_cols.append(q_j)

    q = Tensor.stack(q_cols, dim=1)
    r = q.T @ a

    # To match numpy's convention, ensure diagonal elements of R are non-negative.
    # The QR decomposition is not unique; we can flip the signs of corresponding
    # columns in Q and rows in R.
    diag_r = r.flatten()[:: r.shape[1] + 1][:k]
    signs = (diag_r < 0).cast(a.dtype) * -2 + 1
    q = q * signs
    r = signs.reshape(-1, 1) * r

    return q, r
