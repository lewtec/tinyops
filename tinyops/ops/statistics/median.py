from tinygrad import Tensor


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
    if axis < 0:
        axis += number_of_dimensions

    if axis < 0 or axis >= number_of_dimensions:
        raise ValueError(f"Axis {axis} out of bounds for array of dimension {number_of_dimensions}")

    # Move target axis to last position for sorting
    if axis != number_of_dimensions - 1:
        permutation = [i for i in range(number_of_dimensions) if i != axis] + [axis]
        tensor = tensor.permute(permutation)

    sorted_tensor, _ = tensor.sort()
    length = sorted_tensor.shape[-1]

    if length % 2 == 1:
        middle = (length - 1) // 2
        result = sorted_tensor[..., middle : middle + 1]
    else:
        lower_middle = length // 2 - 1
        upper_middle = length // 2
        lower_values = sorted_tensor[..., lower_middle : lower_middle + 1]
        upper_values = sorted_tensor[..., upper_middle : upper_middle + 1]
        result = (lower_values + upper_values) / 2

    if keep_dimensions:
        if axis != number_of_dimensions - 1:
            inverse_permutation = list(range(axis)) + [number_of_dimensions - 1] + list(range(axis, number_of_dimensions - 1))
            result = result.permute(inverse_permutation)
    else:
        result = result.squeeze(-1)

    return result
