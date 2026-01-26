from tinygrad import Tensor

from tinyops.ml.precision import precision
from tinyops.ml.recall import recall


def f1(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Calculates the F1 score.

    Args:
      y_true: Ground truth labels.
      y_pred: Predicted labels.

    Returns:
      The F1 score.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / Tensor.where(p + r == 0, 1, p + r)
