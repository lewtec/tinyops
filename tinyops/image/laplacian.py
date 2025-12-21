from tinygrad import Tensor
from ._utils import _apply_3x3_kernel

def laplacian(x: Tensor, ksize: int = 1) -> Tensor:
  """
  Calculates the Laplacian of an image.

  Args:
    x: Input tensor of shape (H, W).
    ksize: Size of the Laplacian kernel. Currently, only ksize=1 is supported.

  Returns:
    Output tensor.
  """
  if ksize != 1:
    raise NotImplementedError("Only ksize=1 is currently supported")

  # For ksize=1, OpenCV uses this specific 3x3 kernel.
  kernel = Tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=x.dtype)

  return _apply_3x3_kernel(x, kernel)
