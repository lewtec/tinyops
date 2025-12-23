from enum import Enum
from functools import partial
from tinygrad import Tensor

def pad_constant(x: Tensor, padding, fill) -> Tensor:
  if isinstance(padding, int):
    p_left = p_right = p_top = p_bottom = padding
  elif isinstance(padding, tuple) and len(padding) == 2:
    p_left = p_right = padding[0]
    p_top = p_bottom = padding[1]
  elif isinstance(padding, tuple) and len(padding) == 4:
    p_left, p_top, p_right, p_bottom = padding
  else:
    raise ValueError("Padding must be an int or a tuple of length 2 or 4.")

  pad_widths = ((p_top, p_bottom), (p_left, p_right))
  if x.ndim == 3:
    pad_widths += ((0, 0),)

  return x.pad(pad_widths, value=fill)

def pad_not_implemented(x: Tensor, padding, fill=None) -> Tensor:
  raise NotImplementedError("This padding mode is not yet implemented.")

class PaddingMode(Enum):
  CONSTANT = (partial(pad_constant),)
  REFLECT = (partial(pad_not_implemented),)
  REPLICATE = (partial(pad_not_implemented),)
  CIRCULAR = (partial(pad_not_implemented),)

  def __call__(self, *args, **kwargs):
    return self.value[0](*args, **kwargs)

def pad(x: Tensor, padding, fill=0, padding_mode="constant") -> Tensor:
  """
  Pads an image.

  Args:
    x: Input image tensor (H, W, C) or (H, W).
    padding: Padding on each border.
    fill: Value for constant padding.
    padding_mode: Type of padding. "constant", "reflect", "replicate" or "circular".

  Returns:
    The padded image tensor.
  """
  if isinstance(padding_mode, str):
    try:
      mode = PaddingMode[padding_mode.upper()]
    except KeyError:
      raise ValueError(f"Padding mode '{padding_mode}' is not supported.")
  elif isinstance(padding_mode, PaddingMode):
    mode = padding_mode
  else:
    raise TypeError(f"Invalid type for padding_mode: {type(padding_mode)}")

  return mode(x, padding, fill=fill)
