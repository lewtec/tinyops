from tinygrad import Tensor
from enum import Enum
from tinyops.image.dilate import dilate
from tinyops.image.erode import erode

class MorphTypes(Enum):
    MORPH_OPEN = 1
    MORPH_CLOSE = 2
    MORPH_GRADIENT = 3
    MORPH_TOPHAT = 4
    MORPH_BLACKHAT = 5

def morphology(x: Tensor, kernel: Tensor, op: MorphTypes, iterations: int = 1) -> Tensor:
    """
    Performs advanced morphological transformations.
    Reference: https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f
    """
    if op == MorphTypes.MORPH_OPEN:
        return dilate(erode(x, kernel, iterations), kernel, iterations)
    elif op == MorphTypes.MORPH_CLOSE:
        return erode(dilate(x, kernel, iterations), kernel, iterations)
    elif op == MorphTypes.MORPH_GRADIENT:
        return dilate(x, kernel, iterations) - erode(x, kernel, iterations)
    elif op == MorphTypes.MORPH_TOPHAT:
        return x - dilate(erode(x, kernel, iterations), kernel, iterations)
    elif op == MorphTypes.MORPH_BLACKHAT:
        return erode(dilate(x, kernel, iterations), kernel, iterations) - x
    else:
        raise ValueError(f"Unsupported morphology operation: {op}")
