from tinygrad import Tensor
from typing import Optional

def cov(m: Tensor, y: Optional[Tensor] = None, rowvar: bool = True, ddof: int = 1) -> Tensor:
    """
    Estimate a covariance matrix, given data and weights.
    """
    if y is not None:
        m = Tensor.stack([m, y])

    if not rowvar and m.shape[0] != 1:
        m = m.permute(1, 0)

    if m.shape[0] == 0:
        return Tensor([])

    if m.ndim > 2:
        raise ValueError("m has more than 2 dimensions")

    X = m
    X_mean = X.mean(axis=1, keepdim=True)
    X_centered = X - X_mean

    N = X.shape[1]

    # tinygrad's var has ddof as `correction`
    # but numpy's cov uses N - ddof in the denominator.
    # So we can't directly use .var()
    # We will compute it manually

    if ddof == 0:
        divisor = N
    else:
        divisor = N - ddof

    if divisor == 0:
      # in numpy this returns nan, let's just return a tensor full of nans
      return Tensor.full(X.shape[0], X.shape[0], float("nan"))


    C = (X_centered @ X_centered.T) / divisor

    return C
