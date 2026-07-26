from tinygrad import Tensor


def _move_axis_to_last(tensor: Tensor, axis: int) -> tuple[Tensor, int]:
    """Normalize *axis* and permute so the reduction axis is last.

    Returns:
        ``(permuted_tensor, normalized_axis)`` where *normalized_axis* is the
        non-negative axis index in the original layout.
    """
    number_of_dimensions = len(tensor.shape)
    if axis < 0:
        axis += number_of_dimensions

    if axis < 0 or axis >= number_of_dimensions:
        raise ValueError(f"Axis {axis} out of bounds for array of dimension {number_of_dimensions}")

    if axis != number_of_dimensions - 1:
        permutation = [i for i in range(number_of_dimensions) if i != axis] + [axis]
        tensor = tensor.permute(permutation)

    return tensor, axis


def _median_from_sorted_last_axis(sorted_tensor: Tensor) -> Tensor:
    """Median values from a tensor already sorted on its last axis.

    Odd length: middle element. Even length: mean of the two central values.
    Keeps a trailing size-1 dimension for optional keepdims restore.
    """
    length = sorted_tensor.shape[-1]

    if length % 2 == 1:
        middle = (length - 1) // 2
        return sorted_tensor[..., middle : middle + 1]

    lower_middle = length // 2 - 1
    upper_middle = length // 2
    lower_values = sorted_tensor[..., lower_middle : lower_middle + 1]
    upper_values = sorted_tensor[..., upper_middle : upper_middle + 1]
    return (lower_values + upper_values) / 2


def median(
    tensor: Tensor,
    axis: int | None = None,
    keep_dimensions: bool = False,
) -> Tensor:
    """Compute the median along the specified axis.

    The median is the middle value of a sorted dataset. For even-length data
    the mean of the two central values is returned.

    Args:
        tensor: Input tensor.
        axis: Axis along which the median is computed. None flattens first.
        keep_dimensions: If True, reduced axes are kept as size-one dimensions.

    Returns:
        Tensor containing the median values.
    """
    if axis is None:
        tensor = tensor.flatten()
        axis = 0

    number_of_dimensions = len(tensor.shape)
    tensor, axis = _move_axis_to_last(tensor, axis)
    sorted_tensor, _ = tensor.sort()
    result = _median_from_sorted_last_axis(sorted_tensor)

    if keep_dimensions:
        if axis != number_of_dimensions - 1:
            inverse_permutation = (
                list(range(axis)) + [number_of_dimensions - 1] + list(range(axis, number_of_dimensions - 1))
            )
            result = result.permute(inverse_permutation)
    else:
        result = result.squeeze(-1)

    return result
