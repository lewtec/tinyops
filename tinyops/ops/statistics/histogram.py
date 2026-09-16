from tinygrad import Tensor

from tinyops.ops.statistics.histogram_dd import histogram_dd


def histogram(
    tensor: Tensor,
    number_of_bins: int = 10,
    value_range: tuple[float, float] | None = None,
    compute_density: bool = False,
) -> tuple[Tensor, Tensor]:
    """Compute the histogram of a tensor.

    Args:
        tensor: Input tensor. Flattened before computation.
        number_of_bins: Number of equal-width bins.
        value_range: The (min, max) range of the bins. If None,
            uses the data min and max.
        compute_density: If True, return probability density instead
            of counts.

    Returns:
        A tuple of (counts, bin_edges) where bin_edges has length
        ``number_of_bins + 1``.
    """
    flat = tensor.flatten()
    val_range = [value_range] if value_range is not None else None

    hist, edges = histogram_dd(
        samples=flat,
        number_of_bins=[number_of_bins],
        value_ranges=val_range,
        compute_density=compute_density,
    )
    return hist, edges[0]
