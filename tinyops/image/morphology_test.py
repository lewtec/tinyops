import cv2
import numpy as np
import pytest
from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.image.morphology import morphology, MORPH_OPEN, MORPH_CLOSE, MORPH_GRADIENT, MORPH_TOPHAT, MORPH_BLACKHAT

@pytest.mark.parametrize("op, cv2_op", [
    (MORPH_OPEN, cv2.MORPH_OPEN),
    (MORPH_CLOSE, cv2.MORPH_CLOSE),
    (MORPH_GRADIENT, cv2.MORPH_GRADIENT),
    (MORPH_TOPHAT, cv2.MORPH_TOPHAT),
    (MORPH_BLACKHAT, cv2.MORPH_BLACKHAT),
])
def test_morphology(op, cv2_op):
    # Grayscale image
    img = np.random.rand(10, 10).astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)

    # tinyops implementation
    tinyops_result = morphology(Tensor(img), op, Tensor(kernel))

    # OpenCV implementation
    opencv_result = cv2.morphologyEx(img, cv2_op, kernel)

    assert_close(tinyops_result, opencv_result)

    # Color image
    img_color = np.random.rand(10, 10, 3).astype(np.float32)

    # tinyops implementation
    tinyops_result_color = morphology(Tensor(img_color), op, Tensor(kernel))

    # OpenCV implementation
    opencv_result_color = cv2.morphologyEx(img_color, cv2_op, kernel)

    assert_close(tinyops_result_color, opencv_result_color)
