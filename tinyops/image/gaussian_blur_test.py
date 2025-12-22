import cv2
import numpy as np
import pytest
from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.image.gaussian_blur import gaussian_blur
from tinyops.test_utils import assert_one_kernel

def _get_input_color():
    img = np.random.rand(10, 12, 3).astype(np.float32)
    return Tensor(img).realize(), img

def _get_input_grayscale():
    img = np.random.rand(10, 12).astype(np.float32)
    return Tensor(img).realize(), img

COLOR_INPUTS = [_get_input_color() for _ in range(3)]
GRAYSCALE_INPUTS = [_get_input_grayscale() for _ in range(3)]

@pytest.mark.parametrize("tensor_img, img", COLOR_INPUTS)
@pytest.mark.parametrize("ksize, sigmaX", [((3, 3), 1.0), ((5, 5), 1.5), ((7, 3), 2.0)])
@assert_one_kernel
def test_gaussian_blur_color(tensor_img, img, ksize, sigmaX):
  """Test gaussian_blur for a 3-channel color image."""
  result = gaussian_blur(tensor_img, ksize, sigmaX).realize()
  expected = cv2.GaussianBlur(img, ksize, sigmaX, borderType=cv2.BORDER_CONSTANT)

  assert_close(result, expected, atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("tensor_img, img", GRAYSCALE_INPUTS)
@pytest.mark.parametrize("ksize, sigmaX", [((3, 3), 1.0), ((5, 5), 1.5), ((3, 7), 2.0)])
@assert_one_kernel
def test_gaussian_blur_grayscale(tensor_img, img, ksize, sigmaX):
  """Test gaussian_blur for a single-channel grayscale image."""
  result = gaussian_blur(tensor_img, ksize, sigmaX).realize()
  expected = cv2.GaussianBlur(img, ksize, sigmaX, borderType=cv2.BORDER_CONSTANT)

  assert_close(result, expected, atol=1e-5, rtol=1e-5)
