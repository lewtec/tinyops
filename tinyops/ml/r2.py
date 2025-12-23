from tinygrad import Tensor

def r2(y_true: Tensor, y_pred: Tensor) -> Tensor:
  """
  R2 (coefficient of determination) regression score function.
  """
  numerator = ((y_true - y_pred) ** 2).sum()
  denominator = ((y_true - y_true.mean()) ** 2).sum()

  # Handle the edge case where y_true is constant.
  # If denominator is 0, score is 1.0 if numerator is also 0, otherwise 0.0.
  r2_score = Tensor.where(
    denominator == 0,
    Tensor.where(numerator == 0, 1.0, 0.0),
    1 - numerator / denominator
  )

  return r2_score
