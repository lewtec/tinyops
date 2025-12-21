from tinygrad import Tensor, dtypes
import numpy as np
import cv2

from tinyops.image.erode import erode
from tinyops._core import assert_close

def test_erode_basic():
  img = np.array([
      [0, 0, 0, 0, 0],
      [0, 1, 1, 1, 0],
      [0, 1, 1, 1, 0],
      [0, 1, 1, 1, 0],
      [0, 0, 0, 0, 0]
  ], dtype=np.float32)

  kernel = np.array([
      [0, 1, 0],
      [1, 1, 1],
      [0, 1, 0]
  ], dtype=np.uint8)

  # OpenCV expects uint8 for grayscale morphology
  img_uint8 = (img * 255).astype(np.uint8)

  expected = cv2.erode(img_uint8, kernel, iterations=1)
  expected = expected.astype(np.float32) / 255.0

  result = erode(Tensor(img), Tensor(kernel, dtype=dtypes.uint8))

  assert_close(result, expected)

def test_erode_iterations():
  img = np.array([
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 1, 1, 1, 0, 0],
      [0, 0, 1, 1, 1, 0, 0],
      [0, 0, 1, 1, 1, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
  ], dtype=np.float32)

  kernel = np.ones((3, 3), dtype=np.uint8)

  img_uint8 = (img * 255).astype(np.uint8)

  expected = cv2.erode(img_uint8, kernel, iterations=1)
  expected = expected.astype(np.float32) / 255.0

  result = erode(Tensor(img), Tensor(kernel, dtype=dtypes.uint8), iterations=1)

  assert_close(result, expected)

def test_erode_color():
  img = np.zeros((10, 10, 3), dtype=np.float32)
  img[4:7, 4:7, :] = 1.0
  kernel = np.ones((3, 3), dtype=np.uint8)

  img_uint8 = (img * 255).astype(np.uint8)

  expected = cv2.erode(img_uint8, kernel, iterations=1)
  expected = expected.astype(np.float32) / 255.0

  # Transpose image to (C, H, W) for tinyops
  img_chw = Tensor(img).permute(2, 0, 1)

  result = erode(img_chw, Tensor(kernel, dtype=dtypes.uint8))

  # Transpose result back to (H, W, C) for comparison
  result_hwc = result.permute(1, 2, 0)

  assert_close(result_hwc, expected)
