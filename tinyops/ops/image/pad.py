from enum import Enum

from tinygrad import Tensor


class PaddingMode(Enum):
    """Image padding modes."""

    CONSTANT = "constant"
    REFLECT = "reflect"


def _parse_padding(padding) -> tuple[int, int, int, int]:
    if isinstance(padding, int):
        return padding, padding, padding, padding
    elif isinstance(padding, tuple) and len(padding) == 2:
        return padding[0], padding[1], padding[0], padding[1]
    elif isinstance(padding, tuple) and len(padding) == 4:
        return padding
    else:
        raise ValueError("Padding must be an int or a tuple of length 2 or 4.")


def _pad_constant(image: Tensor, padding: tuple, fill_value: float) -> Tensor:
    left, top, right, bottom = _parse_padding(padding)
    pad_widths = ((top, bottom), (left, right))
    if image.ndim > 2:
        pad_widths += ((0, 0),) * (image.ndim - 2)
    return image.pad(pad_widths, value=fill_value)


def _pad_reflect(image: Tensor, padding: tuple) -> Tensor:
    left, top, right, bottom = _parse_padding(padding)

    # Reflect padding (exclude edge pixel, OpenCV BORDER_REFLECT_101 style)
    if left > 0:
        image = image[:, 1 : left + 1].flip(1).cat(image, dim=1)
    if right > 0:
        image = image.cat(image[:, -right - 1 : -1].flip(1), dim=1)
    if top > 0:
        image = image[1 : top + 1, ...].flip(0).cat(image, dim=0)
    if bottom > 0:
        image = image.cat(image[-bottom - 1 : -1, ...].flip(0), dim=0)

    return image


def pad_image(
    image: Tensor,
    padding: int | tuple[int, ...],
    fill_value: float = 0,
    padding_mode: PaddingMode | str = PaddingMode.CONSTANT,
) -> Tensor:
    """Pad an image with a constant value or reflection.

    Args:
        image: Input image tensor of shape (H, W) or (H, W, C).
        padding: Padding amount. Can be:
            - int: Same padding on all sides.
            - tuple of 2: (horizontal, vertical).
            - tuple of 4: (left, top, right, bottom).
        fill_value: Value for constant padding. Default is 0.
        padding_mode: Padding strategy (CONSTANT or REFLECT).

    Returns:
        Padded image tensor.
    """
    if isinstance(padding_mode, str):
        try:
            padding_mode = PaddingMode(padding_mode.lower())
        except ValueError:
            try:
                padding_mode = PaddingMode[padding_mode.upper()]
            except KeyError as e:
                from tinyops._core import report_error
                report_error(e, f"Invalid padding mode string: {padding_mode}")
                raise ValueError(f"Padding mode '{padding_mode}' is not supported.") from None

    if padding_mode == PaddingMode.CONSTANT:
        return _pad_constant(image, padding, fill_value)
    elif padding_mode == PaddingMode.REFLECT:
        return _pad_reflect(image, padding)
    else:
        raise ValueError(f"Unsupported padding mode: {padding_mode}")
