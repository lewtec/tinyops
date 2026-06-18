from tinygrad import Tensor


def arithmetic_mean(
    tensor: Tensor,
    axis: int | tuple[int, ...] | None = None,
    keep_dimensions: bool = False,
) -> Tensor:
    """Compute the arithmetic mean along the specified axis.

    Args:
        tensor: Input tensor.
        axis: Axis or axes along which the mean is computed.
            None computes over the flattened tensor.
        keep_dimensions: If True, reduced axes are kept as size-one dimensions.

    Returns:
        Tensor containing the mean values.
    """
    return tensor.mean(axis=axis, keepdim=keep_dimensions)
