from tinygrad import Tensor

from tinyops.image._utils import apply_morphological_filter


def erode(x: Tensor, kernel: Tensor) -> Tensor:
    """
    Erodes an image by using a specific structuring element.
    https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaeb1e0c1033e3f6b891a25d0511362aeb
    """
    return apply_morphological_filter(x, kernel, mode="min")
