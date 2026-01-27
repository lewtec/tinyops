from tinygrad import Tensor

from tinyops.image._utils import apply_filter


def box_filter(x: Tensor, ksize: tuple[int, int]) -> Tensor:
  """
  Blurs an image using the box filter. This implementation uses zero-padding,
  which is equivalent to cv2.BORDER_CONSTANT with value 0.

  Args:
    x: Input image tensor (H, W, C) or (H, W).
    ksize: Blurring kernel size.

  Returns:
    Blurred image tensor.
  """
  h, w = ksize

  # Create the box kernel
  kernel_2d = Tensor.ones(h, w, requires_grad=False) / (h * w)

  # apply_filter handles reshaping, padding (defaulting to SAME), and grouped convolution
  return apply_filter(x, kernel_2d, padding_mode='constant')
