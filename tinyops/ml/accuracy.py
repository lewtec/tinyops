from tinygrad import Tensor


def accuracy(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Calculates the accuracy score.

    Args:
      y_true: Ground truth labels.
      y_pred: Predicted labels.

    Returns:
      The accuracy score.
    """
    return (y_true == y_pred).mean()
