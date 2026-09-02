from tinygrad import Tensor


def _resolve_axes_and_permute(tensor: Tensor, axis_1: int, axis_2: int) -> Tensor:
    number_of_dimensions = len(tensor.shape)
    if axis_1 < 0:
        axis_1 += number_of_dimensions
    if axis_2 < 0:
        axis_2 += number_of_dimensions

    if axis_1 == axis_2:
        raise ValueError("axis_1 and axis_2 cannot be the same")

    permutation = [i for i in range(number_of_dimensions) if i != axis_1 and i != axis_2]
    permutation.append(axis_1)
    permutation.append(axis_2)

    return tensor.permute(permutation)


def _calculate_diagonal_slice(rows: int, columns: int, offset: int) -> tuple[int, int, int]:
    if offset >= 0:
        start = offset
        length = min(rows, columns - offset)
    else:
        start = -offset * columns
        length = min(rows + offset, columns)

    if length <= 0:
        return 0, 0, 1

    step = columns + 1
    return start, start + length * step, step


def diagonal(
    tensor: Tensor,
    offset: int = 0,
    axis_1: int = 0,
    axis_2: int = 1,
) -> Tensor:
    """Return specified diagonals of a tensor.

    Args:
        tensor: Input tensor (at least 2D).
        offset: Offset of the diagonal from the main diagonal.
        axis_1: First axis of the 2D sub-arrays.
        axis_2: Second axis of the 2D sub-arrays.

    Returns:
        Tensor containing the requested diagonal elements.
    """
    permuted = _resolve_axes_and_permute(tensor, axis_1, axis_2)
    rows, columns = permuted.shape[-2], permuted.shape[-1]

    start, end, step = _calculate_diagonal_slice(rows, columns, offset)

    flat_shape = list(permuted.shape[:-2]) + [rows * columns]
    flattened = permuted.reshape(tuple(flat_shape))

    return flattened[..., start:end:step]
