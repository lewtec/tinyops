from tinygrad import Tensor


def std(a: Tensor, axis: int | tuple[int, ...] | None = None, ddof: int = 0, keepdims: bool = False) -> Tensor:
    """
    Compute the standard deviation along the specified axis.
    """
    return a.std(axis=axis, correction=ddof, keepdim=keepdims)
