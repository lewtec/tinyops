import cv2
import numpy as np
import pytest
from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.image.erode import erode

@pytest.mark.parametrize("kernel_shape", [(3, 3), (5, 5)])
@pytest.mark.parametrize("kernel_type", ["rect", "cross"])
def test_erode(kernel_shape, kernel_type):
    if kernel_type == "rect":
        kernel = np.ones(kernel_shape, np.uint8)
    elif kernel_type == "cross":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_shape)

    # Grayscale image
    img = np.random.rand(20, 20).astype(np.float32)

    # tinyops implementation
    tinyops_result = erode(Tensor(img), Tensor(kernel))

    # OpenCV implementation
    opencv_result = cv2.erode(img, kernel, iterations=1)

    assert_close(tinyops_result, opencv_result)

    # Color image
    img_color = np.random.rand(20, 20, 3).astype(np.float32)

    # tinyops implementation
    tinyops_result_color = erode(Tensor(img_color), Tensor(kernel))

    # OpenCV implementation
    opencv_result_color = cv2.erode(img_color, kernel, iterations=1)

    assert_close(tinyops_result_color, opencv_result_color)
