from tinygrad import Tensor
from tinyops.stats.percentile import percentile

def robust_scaler(x: Tensor) -> Tensor:
  """
  Scale features using statistics that are robust to outliers.

  This Scaler removes the median and scales the data according to the
  interquartile range (IQR). The IQR is the range between the 1st quartile
  (25th quantile) and the 3rd quartile (75th quantile).

  Args:
    x: The data to scale.

  Returns:
    The scaled data.
  """
  median = percentile(x, 50, axis=0)
  q1 = percentile(x, 25, axis=0)
  q3 = percentile(x, 75, axis=0)
  iqr = q3 - q1

  scale = Tensor.where(iqr == 0, 1.0, iqr)

  return (x - median) / scale
