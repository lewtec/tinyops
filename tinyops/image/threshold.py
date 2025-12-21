from tinygrad import Tensor

# Thresholding types
THRESH_BINARY = 0
THRESH_BINARY_INV = 1
THRESH_TRUNC = 2
THRESH_TOZERO = 3
THRESH_TOZERO_INV = 4

def threshold(src: Tensor, thresh: float, maxval: float, type: int) -> Tensor:
    """
    Applies a fixed-level threshold to a single-channel array.

    Args:
        src: Input array.
        thresh: Threshold value.
        maxval: Maximum value to use with THRESH_BINARY and THRESH_BINARY_INV.
        type: Thresholding type.

    Returns:
        The thresholded array.
    """
    if type == THRESH_BINARY:
        return (src > thresh).where(maxval, 0)
    elif type == THRESH_BINARY_INV:
        return (src > thresh).where(0, maxval)
    elif type == THRESH_TRUNC:
        return (src > thresh).where(thresh, src)
    elif type == THRESH_TOZERO:
        return (src > thresh).where(src, 0)
    elif type == THRESH_TOZERO_INV:
        return (src > thresh).where(0, src)
    else:
        raise ValueError(f"Unsupported thresholding type: {type}")
