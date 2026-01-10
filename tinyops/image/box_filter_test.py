import cv2
import numpy as np
import pytest
from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.image.box_filter import box_filter
from tinyops._core import assert_one_kernel

def _get_input_color():
    img = np.random.rand(10, 12, 3).astype(np.float32)
    return Tensor(img).realize(), img

def _get_input_grayscale():
    img = np.random.rand(10, 12).astype(np.float32)
    return Tensor(img).realize(), img

@pytest.mark.parametrize("ksize", [(3, 3), (4, 4), (5, 5)])
@assert_one_kernel
def test_box_filter_color(ksize):
  """Test box_filter for a 3-channel color image."""
  tensor_img, img = _get_input_color()

  result = box_filter(tensor_img, ksize).realize()
  expected = cv2.boxFilter(img, -1, ksize, borderType=cv2.BORDER_CONSTANT)

  assert_close(result, expected, atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("ksize", [(3, 3), (4, 4), (5, 5)])
@assert_one_kernel
def test_box_filter_grayscale(ksize):
  """Test box_filter for a single-channel grayscale image."""
  tensor_img, img = _get_input_grayscale()

  result = box_filter(tensor_img, ksize).realize()
  expected = cv2.boxFilter(img, -1, ksize, borderType=cv2.BORDER_CONSTANT)

  assert_close(result, expected, atol=1e-5, rtol=1e-5)
