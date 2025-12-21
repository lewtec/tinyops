import cv2
import numpy as np
import pytest
from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.image.gaussian_blur import gaussian_blur

@pytest.mark.parametrize("ksize, sigmaX", [((3, 3), 1.0), ((5, 5), 1.5), ((7, 3), 2.0)])
def test_gaussian_blur_color(ksize, sigmaX):
  """Test gaussian_blur for a 3-channel color image."""
  img = np.random.rand(10, 12, 3).astype(np.float32)
  tensor_img = Tensor(img)

  result = gaussian_blur(tensor_img, ksize, sigmaX)
  expected = cv2.GaussianBlur(img, ksize, sigmaX, borderType=cv2.BORDER_CONSTANT)

  assert_close(result, expected, atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("ksize, sigmaX", [((3, 3), 1.0), ((5, 5), 1.5), ((3, 7), 2.0)])
def test_gaussian_blur_grayscale(ksize, sigmaX):
  """Test gaussian_blur for a single-channel grayscale image."""
  img = np.random.rand(10, 12).astype(np.float32)
  tensor_img = Tensor(img)

  result = gaussian_blur(tensor_img, ksize, sigmaX)
  expected = cv2.GaussianBlur(img, ksize, sigmaX, borderType=cv2.BORDER_CONSTANT)

  assert_close(result, expected, atol=1e-5, rtol=1e-5)
