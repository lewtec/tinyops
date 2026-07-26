import math

from tinygrad import Tensor


def _axis_element_count(tensor: Tensor, axis: int | tuple[int, ...] | None) -> int:
    """Number of elements reduced along *axis* (product of reduced dims)."""
    if axis is None:
        return math.prod(tensor.shape)
    if isinstance(axis, int):
        return tensor.shape[axis]
    return math.prod(tensor.shape[ax] for ax in axis)


def _broadcast_1d_weights(
    weights: Tensor,
    tensor: Tensor,
    axis: int | tuple[int, ...] | None,
) -> Tensor:
    """Reshape length-matching 1D weights onto *axis* for broadcast multiply.

    When *axis* is a single int and *weights* is 1D with matching length,
    expands to a shape of ones with that axis set. Other layouts are returned
    unchanged for the caller's multiply + sum.
    """
    if axis is None or len(weights.shape) != 1 or not isinstance(axis, int):
        return weights

    number_of_dimensions = len(tensor.shape)
    normalized_axis = axis if axis >= 0 else axis + number_of_dimensions
    if weights.shape[0] != tensor.shape[normalized_axis]:
        raise ValueError(
            f"Length of weights ({weights.shape[0]}) not compatible "
            f"with specified axis ({tensor.shape[normalized_axis]})"
        )
    broadcast_shape = [1] * number_of_dimensions
    broadcast_shape[normalized_axis] = weights.shape[0]
    return weights.reshape(tuple(broadcast_shape))


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
            count = _axis_element_count(tensor, axis)
            return average, Tensor(count, dtype=tensor.dtype, device=tensor.device)
        return average

    adjusted_weights = _broadcast_1d_weights(weights, tensor, axis)
    sum_of_weights = adjusted_weights.sum(axis=axis)
    average = (tensor * adjusted_weights).sum(axis=axis) / sum_of_weights

    if return_sum_of_weights:
        return average, sum_of_weights
    return average
