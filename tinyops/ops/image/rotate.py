from enum import Enum

from tinygrad import Tensor


class RotationAngle(Enum):
    """Fixed rotation angles."""
    CLOCKWISE_90 = "clockwise_90"
    HALF_TURN = "half_turn"
    COUNTERCLOCKWISE_90 = "counterclockwise_90"


def rotate_image(image: Tensor, angle: RotationAngle) -> Tensor:
    """Rotate an image by a fixed angle (multiples of 90 degrees).

    Args:
        image: Input image tensor (H, W) or (H, W, C).
        angle: Rotation angle.

    Returns:
        Rotated image tensor.
    """
    if angle == RotationAngle.CLOCKWISE_90:
        if image.ndim == 2:
            return image.permute(1, 0).flip(1)
        elif image.ndim == 3:
            return image.permute(1, 0, 2).flip(1)
        else:
            raise ValueError(f"Unsupported tensor rank: {image.ndim}")
    elif angle == RotationAngle.HALF_TURN:
        return image.flip((0, 1))
    elif angle == RotationAngle.COUNTERCLOCKWISE_90:
        if image.ndim == 2:
            return image.permute(1, 0).flip(0)
        elif image.ndim == 3:
            return image.permute(1, 0, 2).flip(0)
        else:
            raise ValueError(f"Unsupported tensor rank: {image.ndim}")
    else:
        raise ValueError(f"Invalid rotation angle: {angle}")
