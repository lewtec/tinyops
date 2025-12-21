import cv2
import numpy as np
from tinygrad import Tensor
from tinyops.image.laplacian import laplacian
from tinyops._core import assert_close

def test_laplacian():
  img = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
  img_tensor = Tensor(img.astype(np.float32))

  laplacian_img = laplacian(img_tensor, ksize=1)
  expected_img = cv2.Laplacian(img, cv2.CV_32F, ksize=1)
  assert_close(laplacian_img, Tensor(expected_img))
