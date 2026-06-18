"""OpenCV 4.x compatibility layer.

Provides cv2-compatible function signatures that delegate to tinyops.ops.
"""

from tinygrad import Tensor

from tinyops.ops.image.box_blur import box_blur as _box_blur
from tinyops.ops.image.color_conversion import ColorSpace, convert_color_space
from tinyops.ops.image.flip import FlipDirection, flip_image
from tinyops.ops.image.gaussian_blur import gaussian_blur as _gaussian_blur
from tinyops.ops.image.histogram_equalization import histogram_equalization
from tinyops.ops.image.laplacian_filter import laplacian_filter
from tinyops.ops.image.morphological_operation import (
    MorphologicalOperation,
    morphological_dilate,
    morphological_erode,
    morphological_operation,
)
from tinyops.ops.image.normalize_intensity import normalize_intensity
from tinyops.ops.image.resize import InterpolationMethod, resize_image
from tinyops.ops.image.rotate import RotationAngle, rotate_image
from tinyops.ops.image.scharr_gradient import scharr_gradient
from tinyops.ops.image.sobel_gradient import GradientDirection, sobel_gradient
from tinyops.ops.image.spatial_filter import spatial_filter as _spatial_filter
from tinyops.ops.image.threshold import ThresholdMethod, apply_threshold

# --- OpenCV constants ---

INTER_NEAREST = 0
INTER_LINEAR = 1

ROTATE_90_CLOCKWISE = 0
ROTATE_180 = 1
ROTATE_90_COUNTERCLOCKWISE = 2

THRESH_BINARY = 0
THRESH_BINARY_INV = 1
THRESH_TRUNC = 2
THRESH_TOZERO = 3
THRESH_TOZERO_INV = 4

MORPH_OPEN = 2
MORPH_CLOSE = 3
MORPH_GRADIENT = 4
MORPH_TOPHAT = 5
MORPH_BLACKHAT = 6

COLOR_BGR2GRAY = 6

NORM_MINMAX = 32

_INTERPOLATION_MAP = {
    INTER_NEAREST: InterpolationMethod.NEAREST_NEIGHBOR,
    INTER_LINEAR: InterpolationMethod.BILINEAR,
}

_ROTATION_MAP = {
    ROTATE_90_CLOCKWISE: RotationAngle.CLOCKWISE_90,
    ROTATE_180: RotationAngle.HALF_TURN,
    ROTATE_90_COUNTERCLOCKWISE: RotationAngle.COUNTERCLOCKWISE_90,
}

_THRESHOLD_MAP = {
    THRESH_BINARY: ThresholdMethod.BINARY,
    THRESH_BINARY_INV: ThresholdMethod.BINARY_INVERSE,
    THRESH_TRUNC: ThresholdMethod.TRUNCATE,
    THRESH_TOZERO: ThresholdMethod.TO_ZERO,
    THRESH_TOZERO_INV: ThresholdMethod.TO_ZERO_INVERSE,
}

_MORPHOLOGY_MAP = {
    MORPH_OPEN: MorphologicalOperation.OPEN,
    MORPH_CLOSE: MorphologicalOperation.CLOSE,
    MORPH_GRADIENT: MorphologicalOperation.GRADIENT,
    MORPH_TOPHAT: MorphologicalOperation.TOP_HAT,
    MORPH_BLACKHAT: MorphologicalOperation.BLACK_HAT,
}


def resize(src: Tensor, dsize: tuple[int, int], interpolation: int = INTER_LINEAR) -> Tensor:
    """Resize an image. dsize is (width, height) as in OpenCV."""
    width, height = dsize
    method = _INTERPOLATION_MAP[interpolation]
    return resize_image(src, target_size=(height, width), method=method)


def rotate(src: Tensor, rotateCode: int) -> Tensor:
    """Rotate an image by 90-degree multiples."""
    return rotate_image(src, _ROTATION_MAP[rotateCode])


def flip(src: Tensor, flipCode: int) -> Tensor:
    """Flip an image.

    flipCode > 0: horizontal, 0: vertical, < 0: both
    """
    if flipCode > 0:
        direction = FlipDirection.HORIZONTAL
    elif flipCode == 0:
        direction = FlipDirection.VERTICAL
    else:
        direction = FlipDirection.BOTH
    return flip_image(src, direction)


def blur(src: Tensor, ksize: tuple[int, int]) -> Tensor:
    """Blur an image using a normalized box filter."""
    return _box_blur(src, kernel_size=ksize)


def boxFilter(src: Tensor, ddepth: int, ksize: tuple[int, int], normalize: bool = True) -> Tensor:
    """Blur an image using a box filter."""
    return _box_blur(src, kernel_size=ksize)


def GaussianBlur(src: Tensor, ksize: tuple[int, int], sigmaX: float, sigmaY: float = 0) -> Tensor:
    """Blur an image using a Gaussian filter."""
    return _gaussian_blur(src, kernel_size=ksize, sigma_x=sigmaX, sigma_y=sigmaY)


def filter2D(src: Tensor, ddepth: int, kernel: Tensor) -> Tensor:
    """Convolve an image with a custom kernel."""
    return _spatial_filter(src, kernel)


def Sobel(src: Tensor, ddepth: int, dx: int, dy: int, ksize: int = 3, scale: float = 1, delta: float = 0) -> Tensor:
    """Compute Sobel derivatives."""
    if dx == 1 and dy == 0:
        direction = GradientDirection.HORIZONTAL
    elif dx == 0 and dy == 1:
        direction = GradientDirection.VERTICAL
    else:
        raise ValueError("Only first-order derivatives (dx=1,dy=0 or dx=0,dy=1) are supported")
    return sobel_gradient(src, direction=direction, kernel_size=ksize, scale=scale, delta=delta)


def Scharr(src: Tensor, ddepth: int, dx: int, dy: int, scale: float = 1, delta: float = 0) -> Tensor:
    """Compute Scharr derivatives."""
    if dx == 1 and dy == 0:
        direction = GradientDirection.HORIZONTAL
    elif dx == 0 and dy == 1:
        direction = GradientDirection.VERTICAL
    else:
        raise ValueError("Only first-order derivatives (dx=1,dy=0 or dx=0,dy=1) are supported")
    return scharr_gradient(src, direction=direction, scale=scale, delta=delta)


def Laplacian(src: Tensor, ddepth: int, ksize: int = 1, scale: float = 1, delta: float = 0) -> Tensor:
    """Compute the Laplacian of an image."""
    return laplacian_filter(src, kernel_size=ksize, scale=scale, delta=delta)


def erode(src: Tensor, kernel: Tensor) -> Tensor:
    """Erode an image."""
    return morphological_erode(src, kernel)


def dilate(src: Tensor, kernel: Tensor) -> Tensor:
    """Dilate an image."""
    return morphological_dilate(src, kernel)


def morphologyEx(src: Tensor, op: int, kernel: Tensor) -> Tensor:
    """Perform advanced morphological transformations."""
    return morphological_operation(src, _MORPHOLOGY_MAP[op], kernel)


def threshold(src: Tensor, thresh: float, maxval: float, type: int) -> tuple[float, Tensor]:
    """Apply a fixed-level threshold. Returns (thresh_used, thresholded_image)."""
    method = _THRESHOLD_MAP[type]
    result = apply_threshold(src, threshold_value=thresh, maximum_value=maxval, method=method)
    return thresh, result


def cvtColor(src: Tensor, code: int) -> Tensor:
    """Convert image color space."""
    if code == COLOR_BGR2GRAY:
        return convert_color_space(src, ColorSpace.BGR_TO_GRAYSCALE)
    raise NotImplementedError(f"Color conversion code {code} is not supported")


def equalizeHist(src: Tensor) -> Tensor:
    """Equalize the histogram of a grayscale image."""
    return histogram_equalization(src)


def normalize(src: Tensor, dst, alpha: float = 0, beta: float = 255, norm_type: int = NORM_MINMAX) -> Tensor:
    """Normalize an image (NORM_MINMAX only)."""
    if norm_type != NORM_MINMAX:
        raise NotImplementedError("Only NORM_MINMAX is supported")
    return normalize_intensity(src, target_minimum=alpha, target_maximum=beta)