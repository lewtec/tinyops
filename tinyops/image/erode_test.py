import cv2
import numpy as np
import pytest
from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.image.erode import erode
from tinyops.test_utils import assert_one_kernel

def _get_input(shape):
    img = np.random.rand(*shape).astype(np.float32)
    return Tensor(img).realize(), img

def _get_kernel(kernel_shape, kernel_type):
    if kernel_type == "rect":
        kernel = np.ones(kernel_shape, np.uint8)
    elif kernel_type == "cross":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_shape)
    return Tensor(kernel).realize(), kernel

@pytest.mark.parametrize("kernel_shape", [(3, 3), (5, 5)])
@pytest.mark.parametrize("kernel_type", ["rect", "cross"])
@assert_one_kernel
def test_erode_grayscale(kernel_shape, kernel_type):
    tensor_img, img = _get_input((20, 20))
    tensor_kernel, kernel = _get_kernel(kernel_shape, kernel_type)

    tinyops_result = erode(tensor_img, tensor_kernel).realize()
    opencv_result = cv2.erode(img, kernel, iterations=1)

    assert_close(tinyops_result, opencv_result)

@pytest.mark.parametrize("kernel_shape", [(3, 3), (5, 5)])
@pytest.mark.parametrize("kernel_type", ["rect", "cross"])
@assert_one_kernel
def test_erode_color(kernel_shape, kernel_type):
    tensor_img, img = _get_input((20, 20, 3))
    tensor_kernel, kernel = _get_kernel(kernel_shape, kernel_type)

    tinyops_result = erode(tensor_img, tensor_kernel).realize()
    opencv_result = cv2.erode(img, kernel, iterations=1)

    assert_close(tinyops_result, opencv_result)
