import cv2
import numpy as np
from tinygrad import Tensor
from tinyops.image.sobel import sobel
from tinyops._core import assert_close

def test_sobel():
  img = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
  img_tensor = Tensor(img.astype(np.float32))

  # Test dx=1, dy=0
  sobel_x = sobel(img_tensor, dx=1, dy=0)
  expected_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
  assert_close(sobel_x, Tensor(expected_x))

  # Test dx=0, dy=1
  sobel_y = sobel(img_tensor, dx=0, dy=1)
  expected_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
  assert_close(sobel_y, Tensor(expected_y))
