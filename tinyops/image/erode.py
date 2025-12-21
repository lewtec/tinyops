from tinygrad import Tensor
import numpy as np

def erode(x: Tensor, kernel: Tensor, iterations: int = 1) -> Tensor:
  """
  Erodes an image by using a specific structuring element.
  Reference: https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gaeb1e0c1033e3f6b891a25d0511362aeb
  """
  if not all(s % 2 == 1 for s in kernel.shape):
    raise ValueError("Kernel dimensions must be odd")

  in_shape = x.shape
  h, w = in_shape[-2:]

  cy, cx = kernel.shape[0] // 2, kernel.shape[1] // 2

  p_top, p_left = cy, cx
  p_bottom, p_right = kernel.shape[0] - 1 - cy, kernel.shape[1] - 1 - cx

  pad_dims = [(0,0)] * (len(in_shape) - 2) + [(p_top, p_bottom), (p_left, p_right)]

  k_np = kernel.numpy()
  coords = list(zip(*k_np.nonzero()))

  if not coords:
    return Tensor.full(*x.shape, fill_value=float('inf'), dtype=x.dtype)

  y = x
  for _ in range(iterations):
    padded = y.pad(pad_dims, value=float('inf'))

    # Initialize with the first shifted view
    u, v = coords[0]
    dy, dx = int(u) - cy, int(v) - cx
    sy, sx = p_top - dy, p_left - dx
    sl = [slice(None)] * (len(in_shape) - 2) + [slice(sy, sy + h), slice(sx, sx + w)]
    result = padded[tuple(sl)]

    # Lazy min over the rest
    for u, v in coords[1:]:
        dy, dx = int(u) - cy, int(v) - cx
        sy, sx = p_top - dy, p_left - dx
        sl = [slice(None)] * (len(in_shape) - 2) + [slice(sy, sy + h), slice(sx, sx + w)]
        result = result.minimum(padded[tuple(sl)])

    y = result.realize()

  return y
