from enum import Enum

from tinygrad import Tensor


class FlipDirection(Enum):
    """Image flip directions."""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    BOTH = "both"


def flip_image(image: Tensor, direction: FlipDirection) -> Tensor:
    """Flip an image along the specified direction.

    Args:
        image: Input image tensor.
        direction: Flip direction.

    Returns:
        Flipped image tensor.
    """
    if direction == FlipDirection.VERTICAL:
        return image.flip(0)
    elif direction == FlipDirection.HORIZONTAL:
        return image.flip(1)
    elif direction == FlipDirection.BOTH:
        return image.flip((0, 1))
    else:
        raise ValueError(f"Invalid flip direction: {direction}")
