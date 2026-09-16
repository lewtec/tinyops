from tinygrad import Tensor


def peak_to_peak(
    tensor: Tensor,
    axis: int | tuple[int, ...] | None = None,
    keep_dimensions: bool = False,
) -> Tensor:
    """Compute the range of values (maximum minus minimum) along an axis.

    Args:
        tensor: Input tensor.
        axis: Axis or axes along which to compute the range.
        keep_dimensions: If True, reduced axes are kept as size-one dimensions.

    Returns:
        Tensor containing the peak-to-peak range.
    """
    return tensor.max(axis=axis, keepdim=keep_dimensions) - tensor.min(axis=axis, keepdim=keep_dimensions)
