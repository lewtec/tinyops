import math

from tinygrad import Tensor


def weighted_average(
    tensor: Tensor,
    axis: int | tuple[int, ...] | None = None,
    weights: Tensor | None = None,
    return_sum_of_weights: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Compute the weighted average along the specified axis.

    When weights are None, this is equivalent to arithmetic_mean. When weights
    are provided the result is ``sum(tensor * weights) / sum(weights)`` along
    the requested axis.

    Args:
        tensor: Input tensor.
        axis: Axis or axes along which the average is computed.
        weights: Per-element or per-axis weights. Must be broadcastable to
            *tensor* when *axis* is given.
        return_sum_of_weights: If True, return ``(average, sum_of_weights)``.

    Returns:
        The weighted average, or a tuple of (average, sum_of_weights).
    """
    if weights is None:
        average = tensor.mean(axis=axis)
        if return_sum_of_weights:
            if axis is None:
                count = math.prod(tensor.shape)
            elif isinstance(axis, int):
                count = tensor.shape[axis]
            else:
                count = math.prod(tensor.shape[ax] for ax in axis)
            return average, Tensor(count, dtype=tensor.dtype, device=tensor.device)
        return average

    adjusted_weights = weights
    number_of_dimensions = len(tensor.shape)

    if axis is not None and len(adjusted_weights.shape) == 1:
        if isinstance(axis, int):
            normalized_axis = axis if axis >= 0 else axis + number_of_dimensions
            if adjusted_weights.shape[0] != tensor.shape[normalized_axis]:
                raise ValueError(
                    f"Length of weights ({adjusted_weights.shape[0]}) not compatible "
                    f"with specified axis ({tensor.shape[normalized_axis]})"
                )
            broadcast_shape = [1] * number_of_dimensions
            broadcast_shape[normalized_axis] = adjusted_weights.shape[0]
            adjusted_weights = adjusted_weights.reshape(tuple(broadcast_shape))

    sum_of_weights = adjusted_weights.sum(axis=axis)
    product = tensor * adjusted_weights
    average = product.sum(axis=axis) / sum_of_weights

    if return_sum_of_weights:
        return average, sum_of_weights
    return average
