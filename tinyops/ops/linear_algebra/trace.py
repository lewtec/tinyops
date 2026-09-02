from tinygrad import Tensor

from tinyops.ops.linear_algebra.diagonal import diagonal


def trace(
    tensor: Tensor,
    offset: int = 0,
    axis_1: int = 0,
    axis_2: int = 1,
) -> Tensor:
    """Return the sum along diagonals of the array.

    Args:
        tensor: Input tensor (at least 2D).
        offset: Offset of the diagonal from the main diagonal.
        axis_1: First axis of the 2D sub-arrays to take the diagonal from.
        axis_2: Second axis of the 2D sub-arrays.

    Returns:
        Sum of diagonal elements.
    """
    return diagonal(tensor, offset=offset, axis_1=axis_1, axis_2=axis_2).sum(axis=-1)
