from enum import Enum
from functools import partial

from tinygrad import Tensor


def _parse_padding(padding):
    if isinstance(padding, int):
        return padding, padding, padding, padding
    elif isinstance(padding, tuple) and len(padding) == 2:
        return padding[0], padding[1], padding[0], padding[1]
    elif isinstance(padding, tuple) and len(padding) == 4:
        return padding  # left, top, right, bottom
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
        x = x[:, 1 : p_left + 1].flip(1).cat(x, dim=1)
    if p_right > 0:
        x = x.cat(x[:, -p_right - 1 : -1].flip(1), dim=1)

    # Pad Height (Dim 0)
    if p_top > 0:
        x = x[1 : p_top + 1, ...].flip(0).cat(x, dim=0)
    if p_bottom > 0:
        x = x.cat(x[-p_bottom - 1 : -1, ...].flip(0), dim=0)

    return x


def pad_not_implemented(x: Tensor, padding, fill=None) -> Tensor:
    raise NotImplementedError("This padding mode is not yet implemented.")


class PaddingMode(Enum):
    CONSTANT = (partial(pad_constant),)
    REFLECT = (partial(pad_reflect),)
    REPLICATE = (partial(pad_not_implemented),)
    CIRCULAR = (partial(pad_not_implemented),)

    def __call__(self, *args, **kwargs):
        return self.value[0](*args, **kwargs)


def pad(x: Tensor, padding, fill=0, padding_mode="constant") -> Tensor:
    """
    Pads an image with a constant value or reflection.

    Args:
      x: Input image tensor of shape (H, W, C) or (H, W).
      padding: Padding configuration. Can be:
        - int: Same padding on all sides.
        - tuple of 2 ints: (pad_left_right, pad_top_bottom).
        - tuple of 4 ints: (left, top, right, bottom).
      fill: Value to fill when padding_mode is "constant". Default is 0.
      padding_mode: Type of padding. Supported: "constant", "reflect".
        - "constant": Pads with a constant value.
        - "reflect": Pads with reflection of image (OpenCV 101 style), excluding edge pixels.

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
