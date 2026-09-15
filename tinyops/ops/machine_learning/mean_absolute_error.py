from tinygrad import Tensor


def mean_absolute_error(true_values: Tensor, predicted_values: Tensor) -> Tensor:
    """Compute the mean absolute error between predictions and targets.

    Lower is better.

    Args:
        true_values: Ground truth values.
        predicted_values: Predicted values.

    Returns:
        Mean absolute error (scalar).
    """
    return (true_values - predicted_values).abs().mean()
