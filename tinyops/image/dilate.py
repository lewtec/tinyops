from tinygrad import Tensor

from tinyops.image._utils import apply_morphological_filter


def dilate(x: Tensor, kernel: Tensor) -> Tensor:
    """
    Dilates an image by using a specific structuring element.
    https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga4ff0f3318642c4f469d0e11f242f3b6c
    """
    return apply_morphological_filter(x, kernel, mode="max")
