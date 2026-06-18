from tinygrad import Tensor, dtypes


def digitize(
    values: Tensor,
    bin_edges: Tensor,
    right_closed: bool = False,
) -> Tensor:
    """Return indices of the bins to which each value belongs.

    Args:
        values: Input tensor.
        bin_edges: 1D tensor of monotonically increasing bin edges.
        right_closed: If True, intervals include the right edge.

    Returns:
        Tensor of bin indices with the same shape as *values*.
    """
    values_expanded = values.unsqueeze(-1)
    edges_expanded = bin_edges.reshape((1,) * values.ndim + (-1,))

    if right_closed:
        comparison = values_expanded > edges_expanded
    else:
        comparison = values_expanded >= edges_expanded

    return comparison.sum(axis=-1).cast(dtypes.int32)
