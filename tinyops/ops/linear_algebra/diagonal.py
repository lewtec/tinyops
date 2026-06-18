from tinygrad import Tensor


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

    permuted = tensor.permute(permutation)
    rows = permuted.shape[-2]
    columns = permuted.shape[-1]

    if offset >= 0:
        start = offset
        length = min(rows, columns - offset)
    else:
        start = -offset * columns
        length = min(rows + offset, columns)

    if length <= 0:
        start = 0
        end = 0
    else:
        step = columns + 1
        end = start + length * step

    flat_shape = list(permuted.shape[:-2]) + [rows * columns]
    flattened = permuted.reshape(tuple(flat_shape))

    return flattened[..., start:end:(columns + 1)]
