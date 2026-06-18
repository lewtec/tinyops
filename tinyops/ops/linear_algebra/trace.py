from tinygrad import Tensor


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
    number_of_dimensions = len(tensor.shape)
    if axis_1 < 0:
        axis_1 += number_of_dimensions
    if axis_2 < 0:
        axis_2 += number_of_dimensions

    if axis_1 == axis_2:
        raise ValueError("axis_1 and axis_2 cannot be the same")

    permutation = [axis_1, axis_2] + [i for i in range(number_of_dimensions) if i != axis_1 and i != axis_2]
    permuted = tensor.permute(permutation)

    rows = permuted.shape[0]
    columns = permuted.shape[1]

    if offset >= 0:
        if offset >= columns:
            remaining_shape = permuted.shape[2:]
            return Tensor.zeros(*remaining_shape)
        length = min(rows, columns - offset)
        row_indices = Tensor.arange(length)
        column_indices = Tensor.arange(length) + offset
    else:
        if -offset >= rows:
            remaining_shape = permuted.shape[2:]
            return Tensor.zeros(*remaining_shape)
        length = min(rows + offset, columns)
        row_indices = Tensor.arange(length) - offset
        column_indices = Tensor.arange(length)

    return permuted[row_indices, column_indices].sum(axis=0)
