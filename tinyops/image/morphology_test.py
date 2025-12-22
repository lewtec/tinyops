import cv2
import numpy as np
import pytest
from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.image.morphology import morphology, MORPH_OPEN, MORPH_CLOSE, MORPH_GRADIENT, MORPH_TOPHAT, MORPH_BLACKHAT
from tinyops.test_utils import assert_one_kernel

def _get_input(shape):
    img = np.random.rand(*shape).astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)
    return Tensor(img).realize(), Tensor(kernel).realize(), img, kernel

@pytest.mark.parametrize("op, cv2_op", [
    (MORPH_OPEN, cv2.MORPH_OPEN),
    (MORPH_CLOSE, cv2.MORPH_CLOSE),
    (MORPH_GRADIENT, cv2.MORPH_GRADIENT),
    (MORPH_TOPHAT, cv2.MORPH_TOPHAT),
    (MORPH_BLACKHAT, cv2.MORPH_BLACKHAT),
])
@assert_one_kernel
def test_morphology(op, cv2_op):
    # Grayscale image
    tensor_img, tensor_kernel, img, kernel = _get_input((10, 10))

    # tinyops implementation
    tinyops_result = morphology(tensor_img, op, tensor_kernel).realize()

    # OpenCV implementation
    opencv_result = cv2.morphologyEx(img, cv2_op, kernel)

    assert_close(tinyops_result, opencv_result)

@pytest.mark.parametrize("op, cv2_op", [
    (MORPH_OPEN, cv2.MORPH_OPEN),
    (MORPH_CLOSE, cv2.MORPH_CLOSE),
    (MORPH_GRADIENT, cv2.MORPH_GRADIENT),
    (MORPH_TOPHAT, cv2.MORPH_TOPHAT),
    (MORPH_BLACKHAT, cv2.MORPH_BLACKHAT),
])
@assert_one_kernel
def test_morphology_color(op, cv2_op):
    # Color image
    tensor_img, tensor_kernel, img, kernel = _get_input((10, 10, 3))

    # tinyops implementation
    tinyops_result = morphology(tensor_img, op, tensor_kernel).realize()

    # OpenCV implementation
    opencv_result = cv2.morphologyEx(img, cv2_op, kernel)

    assert_close(tinyops_result, opencv_result)
