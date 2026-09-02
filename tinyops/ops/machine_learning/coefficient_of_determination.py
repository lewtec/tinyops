from tinygrad import Tensor


def coefficient_of_determination(true_values: Tensor, predicted_values: Tensor) -> Tensor:
    """Compute R-squared (coefficient of determination).

    Returns 1.0 for perfect prediction, 0.0 when the model performs
    no better than predicting the mean, and negative for worse.
    Higher is better.

    Args:
        true_values: Ground truth values.
        predicted_values: Predicted values.

    Returns:
        R-squared score (scalar).
    """
    residual_sum = ((true_values - predicted_values) ** 2).sum()
    total_sum = ((true_values - true_values.mean()) ** 2).sum()
    return Tensor.where(
        total_sum == 0,
        Tensor.where(residual_sum == 0, 1.0, 0.0),
        1 - residual_sum / total_sum,
    )
