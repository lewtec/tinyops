from enum import Enum
from functools import partial
from tinygrad import Tensor

def threshold_binary(src: Tensor, thresh: float, maxval: float) -> Tensor:
    return (src > thresh).where(maxval, 0)

def threshold_binary_inv(src: Tensor, thresh: float, maxval: float) -> Tensor:
    return (src > thresh).where(0, maxval)

def threshold_trunc(src: Tensor, thresh: float, maxval: float) -> Tensor:
    return (src > thresh).where(thresh, src)

def threshold_tozero(src: Tensor, thresh: float, maxval: float) -> Tensor:
    return (src > thresh).where(src, 0)

def threshold_tozero_inv(src: Tensor, thresh: float, maxval: float) -> Tensor:
    return (src > thresh).where(0, src)

class ThresholdType(Enum):
    BINARY = (partial(threshold_binary),)
    BINARY_INV = (partial(threshold_binary_inv),)
    TRUNC = (partial(threshold_trunc),)
    TOZERO = (partial(threshold_tozero),)
    TOZERO_INV = (partial(threshold_tozero_inv),)

    def __call__(self, *args, **kwargs):
        return self.value[0](*args, **kwargs)

# Backward compatibility constants
THRESH_BINARY = 0
THRESH_BINARY_INV = 1
THRESH_TRUNC = 2
THRESH_TOZERO = 3
THRESH_TOZERO_INV = 4

def threshold(src: Tensor, thresh: float, maxval: float, type: int | ThresholdType) -> Tensor:
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
    if isinstance(type, int):
        mapping = {
            THRESH_BINARY: ThresholdType.BINARY,
            THRESH_BINARY_INV: ThresholdType.BINARY_INV,
            THRESH_TRUNC: ThresholdType.TRUNC,
            THRESH_TOZERO: ThresholdType.TOZERO,
            THRESH_TOZERO_INV: ThresholdType.TOZERO_INV,
        }
        if type in mapping:
            mode = mapping[type]
        else:
            raise ValueError(f"Unsupported thresholding type: {type}")
    elif isinstance(type, ThresholdType):
        mode = type
    else:
        raise TypeError(f"Invalid type for thresholding type: {type(type)}")

    return mode(src, thresh, maxval)
