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
  # apply_filter handles channel repetition
  kernel = Tensor.ones(h, w, requires_grad=False) / (h * w)

  # Calculate padding for 'SAME' output size.
  # OpenCV's anchor for even kernels is at k/2 - 1, which means it pulls pixels more
  # from the bottom/right. This corresponds to larger padding at the top/left.
  pad_top = h // 2
  pad_bottom = (h - 1) // 2
  pad_left = w // 2
  pad_right = (w - 1) // 2
  padding = (pad_left, pad_right, pad_top, pad_bottom)

  return apply_filter(x, kernel, padding=padding, padding_mode='constant')
