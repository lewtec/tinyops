from tinygrad import Tensor


def var(a: Tensor, axis: int | tuple[int, ...] | None = None, ddof: int = 0, keepdims: bool = False) -> Tensor:
    """
    Compute the variance along the specified axis.
    """
    return a.var(axis=axis, correction=ddof, keepdim=keepdims)
