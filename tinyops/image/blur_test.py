import cv2
import numpy as np
import pytest
from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.image.blur import blur

@pytest.mark.parametrize("ksize", [(3, 3), (4, 4), (5, 5)])
def test_blur_color(ksize):
  """Test blur for a 3-channel color image."""
  img = np.random.rand(10, 12, 3).astype(np.float32)
  tensor_img = Tensor(img)

  result = blur(tensor_img, ksize)
  # cv2.blur is an alias for cv2.boxFilter with normalize=True (the default)
  # We need to use BORDER_CONSTANT to match our zero-padding implementation
  expected = cv2.blur(img, ksize, borderType=cv2.BORDER_CONSTANT)

  assert_close(result, expected, atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("ksize", [(3, 3), (4, 4), (5, 5)])
def test_blur_grayscale(ksize):
  """Test blur for a single-channel grayscale image."""
  img = np.random.rand(10, 12).astype(np.float32)
  tensor_img = Tensor(img)

  result = blur(tensor_img, ksize)
  expected = cv2.blur(img, ksize, borderType=cv2.BORDER_CONSTANT)

  assert_close(result, expected, atol=1e-5, rtol=1e-5)
