from enum import IntEnum

from tinygrad import Tensor

from tinyops.image.dilate import dilate
from tinyops.image.erode import erode


class MorphOp(IntEnum):
    OPEN = 0
    CLOSE = 1
    GRADIENT = 2
    TOPHAT = 3
    BLACKHAT = 4


# Backwards compatibility
MORPH_OPEN = MorphOp.OPEN
MORPH_CLOSE = MorphOp.CLOSE
MORPH_GRADIENT = MorphOp.GRADIENT
MORPH_TOPHAT = MorphOp.TOPHAT
MORPH_BLACKHAT = MorphOp.BLACKHAT


def morphology(x: Tensor, op: int | MorphOp, kernel: Tensor) -> Tensor:
    """
    Performs advanced morphological transformations.

    Args:
        x: Input image tensor (2D or 3D).
           Currently supports shapes (H, W) or (H, W, C).
        op: Type of morphological operation.
            - OPEN: Erosion followed by Dilation. Useful for removing small noise.
            - CLOSE: Dilation followed by Erosion. Useful for closing small holes inside foreground objects.
            - GRADIENT: Difference between Dilation and Erosion. Useful for finding outlines.
            - TOPHAT: Difference between input image and Opening. Highlights bright details.
            - BLACKHAT: Difference between Closing and input image. Highlights dark details.
        kernel: Structuring element (2D tensor).

    Returns:
        The processed image tensor with the same shape as `x`.

    Raises:
        ValueError: If `op` is not a valid morphological operation.
    """
    match op:
        case MorphOp.OPEN:
            return dilate(erode(x, kernel), kernel)
        case MorphOp.CLOSE:
            return erode(dilate(x, kernel), kernel)
        case MorphOp.GRADIENT:
            return dilate(x, kernel) - erode(x, kernel)
        case MorphOp.TOPHAT:
            return x - dilate(erode(x, kernel), kernel)
        case MorphOp.BLACKHAT:
            return erode(dilate(x, kernel), kernel) - x
        case _:
            raise ValueError(f"Invalid morphology operation: {op}")
