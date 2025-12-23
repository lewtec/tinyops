from tinygrad import Tensor

def recall(y_true: Tensor, y_pred: Tensor) -> Tensor:
  """
  Calculates the recall score.

  Args:
    y_true: Ground truth labels.
    y_pred: Predicted labels.

  Returns:
    The recall score.
  """
  true_positives = (y_true * y_pred).sum()
  possible_positives = y_true.sum()
  return true_positives / Tensor.where(possible_positives == 0, 1, possible_positives)
