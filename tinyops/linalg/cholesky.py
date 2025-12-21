from tinygrad import Tensor

def cholesky(a: Tensor) -> Tensor:
    """
    Computes the Cholesky decomposition of a symmetric positive-definite matrix.

    Args:
        a: A symmetric positive-definite matrix (n, n).

    Returns:
        The lower triangular Cholesky factor L (n, n).
    """
    n = a.shape[0]
    if n == 0:
        return Tensor.zeros(0, 0, dtype=a.dtype)

    cols = []
    for j in range(n):
        # Use previously computed columns to calculate the new one
        l_prev_cols = Tensor.cat(*cols, dim=1) if cols else Tensor.zeros(n, 0, dtype=a.dtype)

        # Calculate diagonal element l_jj
        s1 = (l_prev_cols[j, :] * l_prev_cols[j, :]).sum()
        l_jj = (a[j, j] - s1).sqrt()

        # Calculate the rest of the column j
        if j < n - 1:
            s2 = l_prev_cols[j + 1:, :] @ l_prev_cols[j, :]
            col_j_rest = (a[j + 1:, j] - s2) / l_jj
        else:
            col_j_rest = Tensor.zeros(0, dtype=a.dtype)

        # Assemble the new column j
        zeros_above = Tensor.zeros(j, dtype=a.dtype)
        l_jj_reshaped = l_jj.reshape(1)
        new_col_j = Tensor.cat(zeros_above, l_jj_reshaped, col_j_rest).realize()
        cols.append(new_col_j.unsqueeze(1))

    return Tensor.cat(*cols, dim=1)
