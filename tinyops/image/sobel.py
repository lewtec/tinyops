from tinygrad import Tensor
from ._utils import _apply_3x3_kernel

def sobel(x: Tensor, dx: int, dy: int, ksize: int = 3) -> Tensor:
  """
  Calculates the Sobel filter.

  Args:
    x: Input tensor of shape (H, W).
    dx: Order of the derivative x.
    dy: Order of the derivative y.
    ksize: Size of the Sobel kernel. Currently, only ksize=3 is supported.

  Returns:
    Output tensor.
  """
  if ksize != 3:
    raise NotImplementedError("Only ksize=3 is currently supported")

  if dx == 1 and dy == 0:
    kernel = Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype)
  elif dx == 0 and dy == 1:
    kernel = Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype)
  else:
    raise ValueError("Only dx=1, dy=0 or dx=0, dy=1 is supported")

  return _apply_3x3_kernel(x, kernel)
