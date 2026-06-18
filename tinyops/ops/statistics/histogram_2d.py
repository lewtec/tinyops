from tinygrad import Tensor, dtypes

from tinyops.ops.statistics.bin_count import bin_count


def histogram_2d(
    x_values: Tensor,
    y_values: Tensor,
    number_of_bins: int | list[int] = 10,
    value_range: list[list[float]] | None = None,
    compute_density: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute a 2D histogram from two data arrays.

    Args:
        x_values: First coordinate values. Flattened before computation.
        y_values: Second coordinate values. Flattened before computation.
        number_of_bins: Number of bins per axis, or [bins_x, bins_y].
        value_range: [[x_min, x_max], [y_min, y_max]] or None.
        compute_density: If True, return probability density.

    Returns:
        A tuple of (histogram, x_edges, y_edges).
    """
    x_flat = x_values.flatten()
    y_flat = y_values.flatten()

    if isinstance(number_of_bins, int):
        bins_x, bins_y = number_of_bins, number_of_bins
    else:
        bins_x, bins_y = number_of_bins[0], number_of_bins[1]

    if value_range is None:
        if x_flat.numel() == 0:
            range_x, range_y = [0.0, 1.0], [0.0, 1.0]
        else:
            range_x = [x_flat.min().item(), x_flat.max().item()]
            range_y = [y_flat.min().item(), y_flat.max().item()]
    else:
        range_x, range_y = list(value_range[0]), list(value_range[1])

    if range_x[0] == range_x[1]:
        range_x[0] -= 0.5
        range_x[1] += 0.5
    if range_y[0] == range_y[1]:
        range_y[0] -= 0.5
        range_y[1] += 0.5

    x_edges = Tensor.linspace(range_x[0], range_x[1], bins_x + 1)
    y_edges = Tensor.linspace(range_y[0], range_y[1], bins_y + 1)

    if x_flat.numel() == 0:
        return Tensor.zeros(bins_x, bins_y), x_edges, y_edges

    width_x = (range_x[1] - range_x[0]) / bins_x
    width_y = (range_y[1] - range_y[0]) / bins_y

    idx_x = ((x_flat - range_x[0]) / width_x).floor().cast(dtypes.int32)
    idx_y = ((y_flat - range_y[0]) / width_y).floor().cast(dtypes.int32)

    idx_x = (x_flat == range_x[1]).where(bins_x - 1, idx_x)
    idx_y = (y_flat == range_y[1]).where(bins_y - 1, idx_y)

    valid_mask = (
        (x_flat >= range_x[0]) & (x_flat <= range_x[1]) &
        (y_flat >= range_y[0]) & (y_flat <= range_y[1])
    )

    total_bins = bins_x * bins_y
    flat_indices = (idx_x * bins_y + idx_y).cast(dtypes.int32)
    final_indices = valid_mask.where(flat_indices, total_bins)

    counts_flat = bin_count(final_indices, minimum_output_length=total_bins + 1)
    histogram_result = counts_flat[:total_bins].reshape(bins_x, bins_y)

    if compute_density:
        area = width_x * width_y
        total = histogram_result.sum()
        if total.item() > 0:
            histogram_result = histogram_result / (total * area)

    return histogram_result, x_edges, y_edges
