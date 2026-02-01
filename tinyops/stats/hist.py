from tinygrad import Tensor, dtypes

from .bincount import bincount


def hist(
    x: Tensor, bins: int = 10, range: tuple[float, float] | None = None, density: bool = False
) -> tuple[Tensor, Tensor]:
    """
    Compute the histogram of a tensor.

    The bins (of equal width) are determined by the range argument or the min/max of the input.
    Values outside the range are ignored.

    Args:
        x: Input tensor. Flattened before computation.
        bins: Number of equal-width bins.
        range: The lower and upper range of the bins. If not provided, range is simply
            ``(x.min(), x.max())``.
        density: If True, the result is the value of the probability *density* function
            at the bin, normalized such that the *integral* over the range is 1.

    Returns:
        A tuple of (hist, bin_edges):
            - hist: The values of the histogram.
            - bin_edges: The bin edges (length(hist) + 1).
    """
    x = x.flatten()

    if range is None:
        if x.numel() == 0:
            min_v, max_v = 0.0, 1.0
        else:
            min_v, max_v = x.min().item(), x.max().item()
    else:
        min_v, max_v = range

    if min_v == max_v:
        min_v -= 0.5
        max_v += 0.5

    edges = Tensor.linspace(min_v, max_v, bins + 1)

    if x.numel() == 0:
        return Tensor.zeros(bins), edges

    width = (max_v - min_v) / bins

    # Calculate indices
    indices = ((x - min_v) / width).floor().cast(dtypes.int32)

    mask_valid = (x >= min_v) & (x <= max_v)

    # Map outliers to bins + 1
    indices = mask_valid.where(indices, bins + 1)

    # Handle the right edge case: x == max_v gives index == bins.
    # We want it to be bins - 1.
    indices = (indices == bins).where(bins - 1, indices)

    cnt = bincount(indices, minlength=bins + 2)
    h = cnt[:bins]

    if density:
        db = width
        h = h / (h.sum() * db)

    return h, edges
