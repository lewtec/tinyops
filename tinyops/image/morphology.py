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
    https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f
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
