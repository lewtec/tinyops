from tinygrad import Tensor


def mean_squared_error(true_values: Tensor, predicted_values: Tensor) -> Tensor:
    """Compute the mean squared error between predictions and targets.

    Lower is better.

    Args:
        true_values: Ground truth values.
        predicted_values: Predicted values.

    Returns:
        Mean squared error (scalar).
    """
    return ((true_values - predicted_values) ** 2).mean()
