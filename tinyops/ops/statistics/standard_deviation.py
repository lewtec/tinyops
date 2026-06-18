from tinygrad import Tensor


def standard_deviation(
    tensor: Tensor,
    axis: int | tuple[int, ...] | None = None,
    degrees_of_freedom: int = 0,
    keep_dimensions: bool = False,
) -> Tensor:
    """Compute the standard deviation along the specified axis.

    Args:
        tensor: Input tensor.
        axis: Axis or axes along which the standard deviation is computed.
        degrees_of_freedom: Delta degrees of freedom. The divisor is
            ``N - degrees_of_freedom`` where N is the number of elements.
            Zero gives population std, one gives sample std.
        keep_dimensions: If True, reduced axes are kept as size-one dimensions.

    Returns:
        Tensor containing the standard deviation values.
    """
    return tensor.std(axis=axis, correction=degrees_of_freedom, keepdim=keep_dimensions)
