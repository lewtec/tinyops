from tinygrad import Tensor
from tinyops.image.dilate import dilate
from tinyops.image.erode import erode

# Define constants for morphology operations to match OpenCV
MORPH_OPEN = 2
MORPH_CLOSE = 3
MORPH_GRADIENT = 4
MORPH_TOPHAT = 5
MORPH_BLACKHAT = 6

def morphology(image: Tensor, op: int, kernel: Tensor) -> Tensor:
    """
    Performs advanced morphological transformations.

    Args:
        image: The input image as a Tensor.
        op: The morphological operation to be performed.
            - MORPH_OPEN: Opening
            - MORPH_CLOSE: Closing
            - MORPH_GRADIENT: Morphological Gradient
            - MORPH_TOPHAT: Top Hat
            - MORPH_BLACKHAT: Black Hat
        kernel: The structuring element.

    Returns:
        The transformed image as a Tensor.
    """
    if op == MORPH_OPEN:
        return dilate(erode(image, kernel), kernel)
    elif op == MORPH_CLOSE:
        return erode(dilate(image, kernel), kernel)
    elif op == MORPH_GRADIENT:
        return dilate(image, kernel) - erode(image, kernel)
    elif op == MORPH_TOPHAT:
        return image - dilate(erode(image, kernel), kernel)
    elif op == MORPH_BLACKHAT:
        return erode(dilate(image, kernel), kernel) - image
    else:
        raise ValueError(f"Unknown morphology operation: {op}")
