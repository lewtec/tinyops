"""Shared helpers for histogram bin-range resolution."""

from tinygrad import Tensor

# When min == max, expand the range symmetrically so a single distinct value
# still occupies a non-zero-width bin (numpy histogram convention).
DEGENERATE_RANGE_HALF_WIDTH = 0.5

# Default bin range used when the data is empty and no explicit range is given.
DEFAULT_EMPTY_DATA_RANGE: tuple[float, float] = (0.0, 1.0)


def resolve_histogram_range(
    values: Tensor,
    value_range: tuple[float, float] | list[float] | None = None,
) -> tuple[float, float]:
    """Resolve the (minimum, maximum) bin range for a 1D histogram axis.

    Args:
        values: Flattened sample values used when *value_range* is None.
        value_range: Optional explicit ``(min, max)``. When None, uses the
            data extrema, or :data:`DEFAULT_EMPTY_DATA_RANGE` for empty data.

    Returns:
        ``(minimum, maximum)`` after expanding a degenerate range so the
        bin width is non-zero.
    """
    if value_range is None:
        if values.numel() == 0:
            minimum_value, maximum_value = DEFAULT_EMPTY_DATA_RANGE
        else:
            minimum_value = float(values.min().item())
            maximum_value = float(values.max().item())
    else:
        minimum_value, maximum_value = float(value_range[0]), float(value_range[1])

    if minimum_value == maximum_value:
        minimum_value -= DEGENERATE_RANGE_HALF_WIDTH
        maximum_value += DEGENERATE_RANGE_HALF_WIDTH

    return minimum_value, maximum_value
