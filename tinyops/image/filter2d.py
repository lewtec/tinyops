from tinygrad import Tensor
from tinyops.image._utils import apply_filter

def filter2d(x: Tensor, kernel: Tensor) -> Tensor:
    """
    Applies a 2D filter to an image, matching cv2.filter2D with default BORDER_REFLECT_101.
    """
    return apply_filter(x, kernel, padding_mode='reflect')
