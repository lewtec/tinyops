from tinygrad import Tensor

def _apply_3x3_kernel(x: Tensor, kernel: Tensor) -> Tensor:
  """
  Applies a 3x3 kernel to an image with BORDER_REFLECT_101 padding.

  Args:
    x: Input tensor of shape (H, W).
    kernel: 3x3 kernel tensor.

  Returns:
    Output tensor.
  """
  if len(x.shape) != 2:
    raise NotImplementedError(f"Only 2D grayscale images (H, W) are supported, but got shape {x.shape}")

  # Manual BORDER_REFLECT_101 padding for ksize=3 (padding=1)
  H, W = x.shape
  left_border = x[:, 1].reshape(H, 1)
  right_border = x[:, -2].reshape(H, 1)
  x_lr_padded = left_border.cat(x, right_border, dim=1)

  top_border = x_lr_padded[1, :].reshape(1, W + 2)
  bottom_border = x_lr_padded[-2, :].reshape(1, W + 2)
  x_padded = top_border.cat(x_lr_padded, bottom_border, dim=0)

  x_conv_ready = x_padded.unsqueeze(0).unsqueeze(0)

  return x_conv_ready.conv2d(kernel.unsqueeze(0).unsqueeze(0)).squeeze()
