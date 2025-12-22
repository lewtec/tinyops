from tinygrad import Tensor

def maxabs_scaler(x: Tensor) -> Tensor:
  """
  Scale each feature by its maximum absolute value.

  This estimator scales and translates each feature individually such
  that the maximal absolute value of each feature in the
  training set will be 1.0. It does not shift/center the data, and
  thus does not destroy any sparsity.

  Args:
    x: The data to scale.

  Returns:
    The scaled data.
  """
  max_abs = x.abs().max(axis=0)
  scale = Tensor.where(max_abs == 0, 1.0, max_abs)
  return x / scale
