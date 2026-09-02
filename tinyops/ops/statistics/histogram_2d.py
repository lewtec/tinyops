from tinygrad import Tensor

from tinyops.ops.statistics.histogram_dd import histogram_dd


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
        number_of_bins = [number_of_bins, number_of_bins]

    if x_flat.numel() == 0 and y_flat.numel() == 0:
        samples = Tensor.empty((0, 2), dtype=x_values.dtype, device=x_values.device)
    else:
        samples = Tensor.stack([x_flat, y_flat], dim=1)

    hist, edges = histogram_dd(
        samples=samples,
        number_of_bins=number_of_bins,
        value_ranges=value_range,
        compute_density=compute_density,
    )

    return hist, edges[0], edges[1]
