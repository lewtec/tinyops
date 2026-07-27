from tinygrad import Tensor

from tinyops.ops.statistics.percentile import PercentileMethod, percentile


def quantile(
    tensor: Tensor,
    probabilities: float | list[float] | Tensor,
    axis: int | tuple[int, ...] | None = None,
    keep_dimensions: bool = False,
    method: PercentileMethod = PercentileMethod.LINEAR,
    weights: Tensor | None = None,
) -> Tensor:
    """Compute the q-th quantile along an axis (NumPy-compatible).

    Equivalent to :func:`percentile` with probabilities in ``[0, 1]``.
    """
    if isinstance(probabilities, (int, float)):
        percentages: float | list[float] | Tensor = float(probabilities) * 100.0
    elif isinstance(probabilities, list):
        percentages = [float(value) * 100.0 for value in probabilities]
    elif isinstance(probabilities, Tensor):
        percentages = probabilities * 100.0
    else:
        raise TypeError(f"Unsupported type for probabilities: {type(probabilities)}")

    return percentile(
        tensor,
        percentages,
        axis=axis,
        keep_dimensions=keep_dimensions,
        method=method,
        weights=weights,
    )
