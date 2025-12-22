import cv2
import numpy as np
import pytest
from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.image.blur import blur
from tinyops.test_utils import assert_one_kernel

def _get_input(shape):
    img = np.random.rand(*shape).astype(np.float32)
    return Tensor(img).realize(), img

@pytest.mark.parametrize("ksize", [(3, 3), (4, 4), (5, 5)])
@assert_one_kernel
def test_blur_color(ksize):
  """Test blur for a 3-channel color image."""
  tensor_img, img = _get_input((10, 12, 3))

  result = blur(tensor_img, ksize).realize()
  # cv2.blur is an alias for cv2.boxFilter with normalize=True (the default)
  # We need to use BORDER_CONSTANT to match our zero-padding implementation
  expected = cv2.blur(img, ksize, borderType=cv2.BORDER_CONSTANT)

  assert_close(result, expected, atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("ksize", [(3, 3), (4, 4), (5, 5)])
@assert_one_kernel
def test_blur_grayscale(ksize):
  """Test blur for a single-channel grayscale image."""
  tensor_img, img = _get_input((10, 12))

  result = blur(tensor_img, ksize).realize()
  expected = cv2.blur(img, ksize, borderType=cv2.BORDER_CONSTANT)

  assert_close(result, expected, atol=1e-5, rtol=1e-5)
