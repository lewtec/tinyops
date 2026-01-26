from tinygrad import Tensor


def precision(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Calculates the precision score.

    Args:
      y_true: Ground truth labels.
      y_pred: Predicted labels.

    Returns:
      The precision score.
    """
    true_positives = (y_true * y_pred).sum()
    predicted_positives = y_pred.sum()
    return true_positives / Tensor.where(predicted_positives == 0, 1, predicted_positives)
