from tinygrad import Tensor, dtypes


def _coerce_percentages(
    percentages: float | list[float] | Tensor,
    *,
    device,
    dtype,
) -> tuple[Tensor, bool]:
    """Normalize percentile queries to a 1D tensor and note scalar queries."""
    is_scalar_query = False
    if isinstance(percentages, (int, float)):
        percentages = [float(percentages)]
        is_scalar_query = True
    elif isinstance(percentages, list):
        percentages = [float(p) for p in percentages]

    if not isinstance(percentages, Tensor):
        percentages_tensor = Tensor(percentages, device=device, dtype=dtype)
    else:
        percentages_tensor = percentages.cast(dtype)
        if len(percentages_tensor.shape) == 0:
            percentages_tensor = percentages_tensor.reshape(1)
            is_scalar_query = True
        elif len(percentages_tensor.shape) > 1:
            raise ValueError("percentages must be 1D or scalar")

    return percentages_tensor, is_scalar_query


def _linear_interpolate_sorted(
    sorted_tensor: Tensor,
    percentages_tensor: Tensor,
    number_of_dimensions: int,
) -> Tensor:
    """Linear-interpolate percentile values from a last-axis-sorted tensor."""
    sample_count = sorted_tensor.shape[-1]

    indices = (sample_count - 1) * percentages_tensor / 100.0
    lower_indices = indices.floor()
    upper_indices = indices.ceil()
    fraction = indices - lower_indices

    lower_integer_indices = lower_indices.cast(dtype=dtypes.int32)
    upper_integer_indices = upper_indices.cast(dtype=dtypes.int32)

    expanded_sorted = sorted_tensor.unsqueeze(0).expand(
        [percentages_tensor.shape[0]] + list(sorted_tensor.shape)
    )

    target_shape = [percentages_tensor.shape[0]] + list(sorted_tensor.shape[:-1]) + [1]
    lower_expanded = lower_integer_indices.reshape(
        (percentages_tensor.shape[0],) + (1,) * (number_of_dimensions - 1) + (1,)
    ).expand(target_shape)
    upper_expanded = upper_integer_indices.reshape(
        (percentages_tensor.shape[0],) + (1,) * (number_of_dimensions - 1) + (1,)
    ).expand(target_shape)

    lower_values = expanded_sorted.gather(-1, lower_expanded).squeeze(-1)
    upper_values = expanded_sorted.gather(-1, upper_expanded).squeeze(-1)

    fraction = fraction.reshape((percentages_tensor.shape[0],) + (1,) * (number_of_dimensions - 1))
    return lower_values + (upper_values - lower_values) * fraction


def percentile(
    tensor: Tensor,
    percentages: float | list[float] | Tensor,
    axis: int | None = None,
    keep_dimensions: bool = False,
) -> Tensor:
    """Compute the q-th percentile of the data along the specified axis.

    Uses linear interpolation between data points.

    Args:
        tensor: Input tensor.
        percentages: Percentile(s) to compute, in range [0, 100].
        axis: Axis along which to compute. None flattens the tensor first.
        keep_dimensions: If True, reduced axes are kept as size-one dimensions.

    Returns:
        Tensor containing the requested percentile values.
    """
    percentages_tensor, is_scalar_query = _coerce_percentages(
        percentages, device=tensor.device, dtype=tensor.dtype
    )

    if axis is None:
        tensor = tensor.flatten()
        axis = 0

    number_of_dimensions = len(tensor.shape)
    if axis < 0:
        axis += number_of_dimensions

    if axis != number_of_dimensions - 1:
        permutation = [i for i in range(number_of_dimensions) if i != axis] + [axis]
        tensor = tensor.permute(permutation)

    sorted_tensor, _ = tensor.sort()
    result = _linear_interpolate_sorted(sorted_tensor, percentages_tensor, number_of_dimensions)

    if is_scalar_query:
        result = result.squeeze(0)
        if keep_dimensions:
            result = result.unsqueeze(axis)
    else:
        if keep_dimensions:
            result = result.unsqueeze(axis + 1)

    return result
