from tinygrad import Tensor
from tinyops.image.box_filter import box_filter

def blur(x: Tensor, ksize: tuple[int, int]) -> Tensor:
  """
  Blurs an image using the box filter. Alias for box_filter.

  Args:
    x: Input image tensor (H, W, C) or (H, W).
    ksize: Blurring kernel size.

  Returns:
    Blurred image tensor.
  """
  return box_filter(x, ksize)
