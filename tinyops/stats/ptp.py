from tinygrad import Tensor


def ptp(a: Tensor, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
    """
    Range of values (maximum - minimum) along an axis.
    """
    return a.max(axis=axis, keepdim=keepdims) - a.min(axis=axis, keepdim=keepdims)
