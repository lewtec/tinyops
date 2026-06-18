from tinygrad import Tensor

from tinyops.ops.statistics.percentile import percentile


def quantile(
    tensor: Tensor,
    probabilities: float | list[float] | Tensor,
    axis: int | None = None,
    keep_dimensions: bool = False,
) -> Tensor:
    """Compute the q-th quantile of the data along the specified axis.

    This is equivalent to percentile but takes probabilities in [0, 1]
    instead of percentages in [0, 100].

    Args:
        tensor: Input tensor.
        probabilities: Quantile(s) to compute, in range [0, 1].
        axis: Axis along which to compute. None flattens the tensor first.
        keep_dimensions: If True, reduced axes are kept as size-one dimensions.

    Returns:
        Tensor containing the requested quantile values.
    """
    if isinstance(probabilities, (int, float)):
        percentages = probabilities * 100.0
    elif isinstance(probabilities, list):
        percentages = [p * 100.0 for p in probabilities]
    elif isinstance(probabilities, Tensor):
        percentages = probabilities * 100.0
    else:
        raise TypeError(f"Unsupported type for probabilities: {type(probabilities)}")

    return percentile(tensor, percentages, axis=axis, keep_dimensions=keep_dimensions)
