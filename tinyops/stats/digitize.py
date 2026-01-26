from tinygrad import Tensor, dtypes


def digitize(x: Tensor, bins: Tensor, right: bool = False) -> Tensor:
    """
    Returns the indices of the bins to which each value in input array belongs.

    Args:
      x: Input tensor.
      bins: 1-D tensor of bins. It has to be sorted, and monotonically increasing.
      right: Indicates whether the intervals include the right or the left bin edge.

    Returns:
      Output tensor of indices, of same shape as x.
    """
    # NOTE: tinygrad has no searchsorted, so this is a simple implementation.
    # https://numpy.org/doc/stable/reference/generated/numpy.digitize.html

    # Reshape for broadcasting.
    x_reshaped = x.unsqueeze(-1)
    bins_reshaped = bins.reshape((1,) * x.ndim + (-1,))

    # Perform the comparison.
    if right:
        # Intervals are (bins[i-1], bins[i]]
        comparison = x_reshaped > bins_reshaped
    else:
        # Intervals are [bins[i-1], bins[i])
        comparison = x_reshaped >= bins_reshaped

    # Summing the boolean tensor along the bins axis gives the indices.
    return comparison.sum(axis=-1).cast(dtypes.int32)
