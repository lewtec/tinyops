import cv2
import numpy as np
import pytest
from tinygrad import Tensor, dtypes
from tinyops._core import assert_close
from tinyops.image.scharr import scharr
from tinyops.test_utils import assert_one_kernel

def _get_input(shape):
    img = np.random.rand(*shape).astype(np.float32) * 255
    img = img.astype(np.uint8)
    return Tensor(img, dtype=dtypes.uint8).realize(), img

@pytest.mark.parametrize("shape", [(10, 20)])
@pytest.mark.parametrize("dx, dy", [(1, 0), (0, 1)])
@pytest.mark.xfail(reason="Scharr uses multiple kernels")
@assert_one_kernel
def test_scharr_grayscale(shape, dx, dy):
    tensor_img, img = _get_input(shape)

    result_tinyops = scharr(tensor_img, dx=dx, dy=dy).realize()
    result_opencv = cv2.Scharr(img, ddepth=cv2.CV_32F, dx=dx, dy=dy)

    assert_close(result_tinyops.cast(dtypes.float32), result_opencv, atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("shape", [(10, 20, 3)])
@pytest.mark.parametrize("dx, dy", [(1, 0), (0, 1)])
@pytest.mark.xfail(reason="Scharr uses multiple kernels")
@assert_one_kernel
def test_scharr_color(shape, dx, dy):
    tensor_img, img = _get_input(shape)

    result_tinyops = scharr(tensor_img, dx=dx, dy=dy).realize()

    result_opencv = np.zeros_like(img, dtype=np.float32)
    for i in range(3):
        result_opencv[:,:,i] = cv2.Scharr(img[:,:,i], ddepth=cv2.CV_32F, dx=dx, dy=dy)

    assert_close(result_tinyops.cast(dtypes.float32), result_opencv, atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("shape", [(5, 10, 20, 3)])
@pytest.mark.parametrize("dx, dy", [(1, 0), (0, 1)])
@pytest.mark.xfail(reason="Scharr uses multiple kernels")
@assert_one_kernel
def test_scharr_batch(shape, dx, dy):
    tensor_img, img = _get_input(shape)

    result_tinyops = scharr(tensor_img, dx=dx, dy=dy).realize()

    result_opencv = np.zeros_like(img, dtype=np.float32)
    for i in range(5):
      for j in range(3):
        result_opencv[i,:,:,j] = cv2.Scharr(img[i,:,:,j], ddepth=cv2.CV_32F, dx=dx, dy=dy)

    assert_close(result_tinyops.cast(dtypes.float32), result_opencv, atol=1e-5, rtol=1e-5)
