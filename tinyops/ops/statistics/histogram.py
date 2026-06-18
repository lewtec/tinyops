from tinygrad import Tensor, dtypes

from tinyops.ops.statistics.bin_count import bin_count


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

    if value_range is None:
        if flat.numel() == 0:
            minimum_value, maximum_value = 0.0, 1.0
        else:
            minimum_value = flat.min().item()
            maximum_value = flat.max().item()
    else:
        minimum_value, maximum_value = value_range

    if minimum_value == maximum_value:
        minimum_value -= 0.5
        maximum_value += 0.5

    edges = Tensor.linspace(minimum_value, maximum_value, number_of_bins + 1)

    if flat.numel() == 0:
        return Tensor.zeros(number_of_bins), edges

    bin_width = (maximum_value - minimum_value) / number_of_bins
    indices = ((flat - minimum_value) / bin_width).floor().cast(dtypes.int32)

    valid_mask = (flat >= minimum_value) & (flat <= maximum_value)
    indices = valid_mask.where(indices, number_of_bins + 1)
    indices = (indices == number_of_bins).where(number_of_bins - 1, indices)

    counts = bin_count(indices, minimum_output_length=number_of_bins + 2)
    histogram_counts = counts[:number_of_bins]

    if compute_density:
        histogram_counts = histogram_counts / (histogram_counts.sum() * bin_width)

    return histogram_counts, edges
