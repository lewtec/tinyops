import cv2
import numpy as np
import pytest
from tinygrad import Tensor, dtypes

from tinyops._core import assert_close
from tinyops.image.sobel import sobel
from tinyops.test_utils import assert_one_kernel

def _get_input_grayscale(h, w):
    img = (np.random.rand(h, w) * 255).astype(np.uint8)
    return Tensor(img).realize(), img

def _get_input_color(h, w, c):
    img = (np.random.rand(h, w, c) * 255).astype(np.uint8)
    return Tensor(img).realize(), img

def _get_input_batch(n, h, w, c):
    img_batch = (np.random.rand(n, h, w, c) * 255).astype(np.uint8)
    return Tensor(img_batch).realize(), img_batch

@pytest.mark.parametrize("dx, dy", [(1, 0), (0, 1)])
@pytest.mark.parametrize("h, w", [(10, 12)])
@assert_one_kernel
def test_sobel_grayscale(h, w, dx, dy):
    """Test Sobel for a single-channel grayscale image."""
    tensor_img, img = _get_input_grayscale(h, w)

    # tinyops implementation
    result = sobel(tensor_img, dx=dx, dy=dy, ksize=3).realize()

    # OpenCV reference: Specify CV_32F to avoid overflow and for accurate comparison
    expected = cv2.Sobel(img, cv2.CV_32F, dx, dy, ksize=3, borderType=cv2.BORDER_CONSTANT)

    assert result.dtype == dtypes.float32
    assert_close(result, expected, atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("dx, dy", [(1, 0), (0, 1)])
@pytest.mark.parametrize("h, w, c", [(10, 12, 3)])
@assert_one_kernel
def test_sobel_color(h, w, c, dx, dy):
    """Test Sobel for a 3-channel color image."""
    tensor_img, img = _get_input_color(h, w, c)

    result = sobel(tensor_img, dx=dx, dy=dy, ksize=3).realize()
    expected = cv2.Sobel(img, cv2.CV_32F, dx, dy, ksize=3, borderType=cv2.BORDER_CONSTANT)

    assert result.dtype == dtypes.float32
    assert_close(result, expected, atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("dx, dy", [(1, 0), (0, 1)])
@pytest.mark.parametrize("n, h, w, c", [(2, 10, 12, 3)])
@assert_one_kernel
def test_sobel_batch(n, h, w, c, dx, dy):
    """Test Sobel for a batch of images."""
    tensor_batch, img_batch = _get_input_batch(n, h, w, c)

    result = sobel(tensor_batch, dx=dx, dy=dy, ksize=3).realize()

    # OpenCV doesn't support batch processing directly, so we loop
    expected_batch = np.array([
        cv2.Sobel(img, cv2.CV_32F, dx, dy, ksize=3, borderType=cv2.BORDER_CONSTANT)
        for img in img_batch
    ])

    assert result.dtype == dtypes.float32
    assert_close(result, expected_batch, atol=1e-5, rtol=1e-5)
