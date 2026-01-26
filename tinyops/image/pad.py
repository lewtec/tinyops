from enum import Enum
from functools import partial

from tinygrad import Tensor

def _parse_padding(padding):
  if isinstance(padding, int):
    return padding, padding, padding, padding
  elif isinstance(padding, tuple) and len(padding) == 2:
    return padding[0], padding[1], padding[0], padding[1]
  elif isinstance(padding, tuple) and len(padding) == 4:
    return padding # left, top, right, bottom
  else:
    raise ValueError("Padding must be an int or a tuple of length 2 or 4.")

def pad_constant(x: Tensor, padding, fill) -> Tensor:
  p_left, p_top, p_right, p_bottom = _parse_padding(padding)

  pad_widths = ((p_top, p_bottom), (p_left, p_right))
  if x.ndim == 3:
    pad_widths += ((0, 0),)

  return x.pad(pad_widths, value=fill)

def pad_reflect(x: Tensor, padding, fill=None) -> Tensor:
  p_left, p_top, p_right, p_bottom = _parse_padding(padding)

  # Reflect padding 101 (OpenCV default)
  # Exclude the edge pixel

  # Pad Width (Dim 1)
  if p_left > 0:
    x = x[:, 1:p_left+1].flip(1).cat(x, dim=1)
  if p_right > 0:
    x = x.cat(x[:, -p_right-1:-1].flip(1), dim=1)

  # Pad Height (Dim 0)
  if p_top > 0:
    x = x[1:p_top+1, ...].flip(0).cat(x, dim=0)
  if p_bottom > 0:
    x = x.cat(x[-p_bottom-1:-1, ...].flip(0), dim=0)

  return x

def pad_not_implemented(x: Tensor, padding, fill=None) -> Tensor:
    raise NotImplementedError("This padding mode is not yet implemented.")


class PaddingMode(Enum):
  CONSTANT = (partial(pad_constant),)
  REFLECT = (partial(pad_reflect),)
  REPLICATE = (partial(pad_not_implemented),)
  CIRCULAR = (partial(pad_not_implemented),)


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
            raise ValueError(f"Padding mode '{padding_mode}' is not supported.") from None
    elif isinstance(padding_mode, PaddingMode):
        mode = padding_mode
    else:
        raise TypeError(f"Invalid type for padding_mode: {type(padding_mode)}")

    return mode(x, padding, fill=fill)
