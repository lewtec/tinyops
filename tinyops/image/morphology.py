from tinygrad import Tensor
from tinyops.image.dilate import dilate
from tinyops.image.erode import erode

MORPH_OPEN = 0
MORPH_CLOSE = 1
MORPH_GRADIENT = 2
MORPH_TOPHAT = 3
MORPH_BLACKHAT = 4

def morphology(x: Tensor, op: int, kernel: Tensor) -> Tensor:
    """
    Performs advanced morphological transformations.
    https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f
    """
    match op:
        case 0: # MORPH_OPEN
            return dilate(erode(x, kernel), kernel)
        case 1: # MORPH_CLOSE
            return erode(dilate(x, kernel), kernel)
        case 2: # MORPH_GRADIENT
            return dilate(x, kernel) - erode(x, kernel)
        case 3: # MORPH_TOPHAT
            return x - dilate(erode(x, kernel), kernel)
        case 4: # MORPH_BLACKHAT
            return erode(dilate(x, kernel), kernel) - x
        case _:
            raise ValueError(f"Invalid morphology operation: {op}")
