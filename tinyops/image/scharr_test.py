import cv2
import numpy as np
from tinygrad import Tensor
from tinyops.image.scharr import scharr
from tinyops._core import assert_close

def test_scharr():
  img = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
  img_tensor = Tensor(img.astype(np.float32))

  # Test dx=1, dy=0
  scharr_x = scharr(img_tensor, dx=1, dy=0)
  expected_x = cv2.Scharr(img, cv2.CV_32F, 1, 0)
  assert_close(scharr_x, Tensor(expected_x))

  # Test dx=0, dy=1
  scharr_y = scharr(img_tensor, dx=0, dy=1)
  expected_y = cv2.Scharr(img, cv2.CV_32F, 0, 1)
  assert_close(scharr_y, Tensor(expected_y))
