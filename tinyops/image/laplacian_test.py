import cv2
import numpy as np
import pytest
from tinygrad import Tensor, dtypes
from tinyops._core import assert_close
from tinyops.image.laplacian import laplacian
from tinyops.test_utils import assert_one_kernel

def _get_input(shape):
    img = np.random.rand(*shape).astype(np.float32) * 255
    img = img.astype(np.uint8)
    return Tensor(img, dtype=dtypes.uint8).realize(), img

@assert_one_kernel
def test_laplacian_ksize_1():
    tensor_img, img = _get_input((10, 20))

    result_tinyops = laplacian(tensor_img, ksize=1).realize()
    result_opencv = cv2.Laplacian(img, ddepth=cv2.CV_32F, ksize=1)

    assert_close(result_tinyops, result_opencv, atol=1e-5, rtol=1e-5)

@assert_one_kernel
def test_laplacian_color():
    tensor_img, img = _get_input((10, 20, 3))

    result_tinyops = laplacian(tensor_img, ksize=1).realize()

    result_opencv = np.zeros_like(img, dtype=np.float32)
    for i in range(3):
        result_opencv[:,:,i] = cv2.Laplacian(img[:,:,i], ddepth=cv2.CV_32F, ksize=1)

    assert_close(result_tinyops, result_opencv, atol=1e-5, rtol=1e-5)

@assert_one_kernel
def test_laplacian_batch():
    tensor_img, img = _get_input((5, 10, 20, 3))

    result_tinyops = laplacian(tensor_img, ksize=1).realize()

    result_opencv = np.zeros_like(img, dtype=np.float32)
    for i in range(5):
      for j in range(3):
        result_opencv[i,:,:,j] = cv2.Laplacian(img[i,:,:,j], ddepth=cv2.CV_32F, ksize=1)

    assert_close(result_tinyops, result_opencv, atol=1e-5, rtol=1e-5)
