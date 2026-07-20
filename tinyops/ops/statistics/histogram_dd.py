"""N-dimensional histogram (numpy.histogramdd equivalent)."""

from collections.abc import Sequence

from tinygrad import Tensor, dtypes

from tinyops.ops.statistics._histogram import resolve_histogram_range
from tinyops.ops.statistics.bin_count import bin_count


def histogram_dd(
    samples: Tensor,
    number_of_bins: int | Sequence[int] = 10,
    value_ranges: Sequence[tuple[float, float] | list[float]] | None = None,
    compute_density: bool = False,
    weights: Tensor | None = None,
) -> tuple[Tensor, list[Tensor]]:
    """Compute an N-dimensional histogram over the columns of *samples*.

    Args:
        samples: Sample coordinates of shape ``(n_samples, n_dimensions)``.
            A 1D tensor of shape ``(n_samples,)`` is treated as one dimension.
        number_of_bins: Shared bin count, or one integer per dimension.
        value_ranges: Optional ``(min, max)`` per dimension. When None, each
            dimension uses the corresponding column extrema (with the same
            degenerate-range expansion as one-dimensional histograms).
        compute_density: If True, normalize so the integral over the bin
            volumes is 1 (probability density).
        weights: Optional per-sample weights of shape ``(n_samples,)``.

    Returns:
        ``(histogram, edges)`` where *histogram* has one axis per dimension
        and *edges* is a list of bin-edge tensors (length ``bins_d + 1`` each).

    Raises:
        ValueError: On invalid rank, bin counts, ranges, or weight shapes.
    """
    if samples.ndim == 1:
        sample_matrix = samples.unsqueeze(1)
    elif samples.ndim == 2:
        sample_matrix = samples
    else:
        raise ValueError(
            f"samples must be 1D or 2D (n_samples, n_dimensions), got shape {samples.shape}"
        )

    sample_count, dimension_count = sample_matrix.shape

    if isinstance(number_of_bins, int):
        bins_per_dimension = [number_of_bins] * dimension_count
    else:
        bins_per_dimension = [int(bin_count_value) for bin_count_value in number_of_bins]
        if len(bins_per_dimension) != dimension_count:
            raise ValueError(
                f"number_of_bins length {len(bins_per_dimension)} must match "
                f"dimension count {dimension_count}"
            )

    if any(bin_count_value < 1 for bin_count_value in bins_per_dimension):
        raise ValueError(f"each bin count must be >= 1, got {bins_per_dimension}")

    if value_ranges is not None and len(value_ranges) != dimension_count:
        raise ValueError(
            f"value_ranges length {len(value_ranges)} must match dimension count {dimension_count}"
        )

    if weights is not None:
        if weights.ndim != 1 or weights.shape[0] != sample_count:
            raise ValueError(
                f"weights must have shape ({sample_count},), got {weights.shape}"
            )

    resolved_ranges: list[tuple[float, float]] = []
    edges: list[Tensor] = []
    bin_widths: list[float] = []

    for dimension_index in range(dimension_count):
        column = sample_matrix[:, dimension_index]
        explicit_range = None if value_ranges is None else value_ranges[dimension_index]
        minimum_value, maximum_value = resolve_histogram_range(column, explicit_range)
        resolved_ranges.append((minimum_value, maximum_value))
        bin_count_for_axis = bins_per_dimension[dimension_index]
        edges.append(Tensor.linspace(minimum_value, maximum_value, bin_count_for_axis + 1))
        bin_widths.append((maximum_value - minimum_value) / bin_count_for_axis)

    total_bins = 1
    for bin_count_for_axis in bins_per_dimension:
        total_bins *= bin_count_for_axis

    if sample_count == 0:
        return Tensor.zeros(*bins_per_dimension), edges

    flat_indices = Tensor.zeros(sample_count, dtype=dtypes.int32)
    valid_mask = Tensor.ones(sample_count, dtype=dtypes.bool)
    stride = 1

    # Build row-major flat indices: (...((i0 * b1 + i1) * b2 + i2) ...)
    # Process dimensions from last to first so the last axis varies fastest
    # (matches numpy.histogramdd / C-order reshape).
    for dimension_index in range(dimension_count - 1, -1, -1):
        column = sample_matrix[:, dimension_index]
        minimum_value, maximum_value = resolved_ranges[dimension_index]
        bin_count_for_axis = bins_per_dimension[dimension_index]
        bin_width = bin_widths[dimension_index]

        indices = ((column - minimum_value) / bin_width).floor().cast(dtypes.int32)
        indices = (column == maximum_value).where(bin_count_for_axis - 1, indices)

        in_range = (column >= minimum_value) & (column <= maximum_value)
        valid_mask = valid_mask & in_range

        flat_indices = flat_indices + indices * stride
        stride *= bin_count_for_axis

    final_indices = valid_mask.where(flat_indices.cast(dtypes.int32), total_bins)

    if weights is None:
        counts_flat = bin_count(final_indices, minimum_output_length=total_bins + 1)
    else:
        # Samples outside the range contribute zero by mapping to the overflow bin.
        safe_weights = valid_mask.where(weights, 0.0)
        counts_flat = bin_count(
            final_indices,
            weights=safe_weights,
            minimum_output_length=total_bins + 1,
        )

    histogram_result = counts_flat[:total_bins].reshape(*bins_per_dimension)

    if compute_density:
        volume = 1.0
        for bin_width in bin_widths:
            volume *= bin_width
        total = histogram_result.sum()
        if float(total.item()) > 0.0:
            histogram_result = histogram_result / (total * volume)

    return histogram_result, edges
