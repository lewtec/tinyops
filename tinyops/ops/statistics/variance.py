from tinygrad import Tensor


def variance(
    tensor: Tensor,
    axis: int | tuple[int, ...] | None = None,
    degrees_of_freedom: int = 0,
    keep_dimensions: bool = False,
) -> Tensor:
    """Compute the variance along the specified axis.

    Args:
        tensor: Input tensor.
        axis: Axis or axes along which the variance is computed.
        degrees_of_freedom: Delta degrees of freedom. The divisor is
            ``N - degrees_of_freedom``.
        keep_dimensions: If True, reduced axes are kept as size-one dimensions.

    Returns:
        Tensor containing the variance values.
    """
    return tensor.var(axis=axis, correction=degrees_of_freedom, keepdim=keep_dimensions)
