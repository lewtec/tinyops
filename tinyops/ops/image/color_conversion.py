from enum import Enum

from tinygrad import Tensor, dtypes


class ColorSpace(Enum):
    """Supported color space conversions."""

    BGR_TO_GRAYSCALE = "bgr_to_grayscale"


# Weights for BGR to grayscale conversion (ITU-R BT.601)
_BLUE_WEIGHT = 0.114
_GREEN_WEIGHT = 0.587
_RED_WEIGHT = 0.299


def convert_color_space(image: Tensor, conversion: ColorSpace) -> Tensor:
    """Convert an image from one color space to another.

    Args:
        image: Input image tensor. Channel layout depends on conversion.
        conversion: Target color space conversion.

    Returns:
        Converted image tensor.

    Raises:
        ValueError: If the image has wrong number of channels.
    """
    if conversion == ColorSpace.BGR_TO_GRAYSCALE:
        if image.shape[-1] != 3:
            raise ValueError("Input image must have 3 channels for BGR to Grayscale conversion.")
        blue = image[..., 0]
        green = image[..., 1]
        red = image[..., 2]
        grayscale = _BLUE_WEIGHT * blue + _GREEN_WEIGHT * green + _RED_WEIGHT * red
        if image.dtype == dtypes.uint8:
            return grayscale.cast(dtypes.uint8)
        return grayscale
    else:
        raise NotImplementedError(f"Color conversion {conversion} not implemented.")
