from tinygrad import Tensor


def mean(a: Tensor, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
    """
    Compute the arithmetic mean along the specified axis.
    """
    return a.mean(axis=axis, keepdim=keepdims)
