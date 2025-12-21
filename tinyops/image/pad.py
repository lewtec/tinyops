from tinygrad import Tensor

def pad(x: Tensor, padding, fill=0, padding_mode="constant") -> Tensor:
  """
  Pads an image.

  Args:
    x: Input image tensor (H, W, C) or (H, W).
    padding: Padding on each border. If a single int is provided this
      is used to pad all borders. If tuple of length 2 is provided this is the padding
      on left/right and top/bottom respectively. If a tuple of length 4 is provided
      this is the padding for the left, top, right and bottom borders respectively.
    fill: Value for constant padding.
    padding_mode: Type of padding. "constant", "reflect", "replicate" or "circular".

  Returns:
    The padded image tensor.
  """
  if padding_mode != "constant":
    raise NotImplementedError(f"Padding mode '{padding_mode}' is not yet implemented.")

  if isinstance(padding, int):
    p_left = p_right = p_top = p_bottom = padding
  elif isinstance(padding, tuple) and len(padding) == 2:
    p_left = p_right = padding[0]
    p_top = p_bottom = padding[1]
  elif isinstance(padding, tuple) and len(padding) == 4:
    p_left, p_top, p_right, p_bottom = padding
  else:
    raise ValueError("Padding must be an int or a tuple of length 2 or 4.")

  # The pad method in tinygrad expects padding for each dimension.
  # For a 2D image (H, W), it's ((top, bottom), (left, right)).
  # For a 3D image (H, W, C), it's ((top, bottom), (left, right), (0, 0)).
  pad_widths = ((p_top, p_bottom), (p_left, p_right))
  if x.ndim == 3:
    pad_widths += ((0, 0),)

  return x.pad(pad_widths, value=fill)
