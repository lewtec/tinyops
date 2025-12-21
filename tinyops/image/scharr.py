from tinygrad import Tensor
from ._utils import _apply_3x3_kernel

def scharr(x: Tensor, dx: int, dy: int) -> Tensor:
  """
  Calculates the Scharr filter.

  Args:
    x: Input tensor of shape (H, W).
    dx: Order of the derivative x.
    dy: Order of the derivative y.

  Returns:
    Output tensor.
  """
  if dx == 1 and dy == 0:
    kernel = Tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=x.dtype)
  elif dx == 0 and dy == 1:
    kernel = Tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=x.dtype)
  else:
    raise ValueError("Only dx=1, dy=0 or dx=0, dy=1 is supported")

  return _apply_3x3_kernel(x, kernel)
