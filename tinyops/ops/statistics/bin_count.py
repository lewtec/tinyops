from tinygrad import Tensor


def bin_count(
    values: Tensor,
    weights: Tensor | None = None,
    minimum_output_length: int = 0,
) -> Tensor:
    """Count occurrences of each non-negative integer value.

    Args:
        values: 1D tensor of non-negative integers.
        weights: Optional weights for each value.
        minimum_output_length: Minimum number of bins in the output.

    Returns:
        Tensor of counts (or weighted counts) for each integer value.

    Raises:
        ValueError: If values is not 1D, is empty with no minimum_output_length,
            or contains negative values.
    """
    if values.ndim != 1:
        raise ValueError("values must be 1D")

    if values.numel() == 0:
        return Tensor.zeros(minimum_output_length)

    minimum_value = int(values.min().item())
    if minimum_value < 0:
        raise ValueError("Input must be non-negative")

    maximum_value = int(values.max().item())
    number_of_classes = max(maximum_value + 1, minimum_output_length)

    one_hot = values.one_hot(number_of_classes)

    if weights is None:
        return one_hot.sum(0)

    if weights.shape != values.shape:
        raise ValueError("weights and values must have the same shape")
    return (one_hot * weights.unsqueeze(-1)).sum(0)
