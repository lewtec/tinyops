import cv2
import numpy as np
from tinygrad import Tensor
from tinyops.image.flip import flip
from tinyops._core import assert_close

def test_flip():
  # Test with a 2D array
  data = np.random.randn(10, 20).astype(np.float32)
  tensor = Tensor(data)

  # Flip around x-axis
  result_0 = flip(tensor, 0)
  expected_0 = cv2.flip(data, 0)
  assert_close(result_0, expected_0)

  # Flip around y-axis
  result_1 = flip(tensor, 1)
  expected_1 = cv2.flip(data, 1)
  assert_close(result_1, expected_1)

  # Flip around both axes
  result_neg_1 = flip(tensor, -1)
  expected_neg_1 = cv2.flip(data, -1)
  assert_close(result_neg_1, expected_neg_1)

  # Test with a 3D array (image)
  data_3d = np.random.randn(10, 20, 3).astype(np.float32)
  tensor_3d = Tensor(data_3d)

  # Flip around x-axis
  result_3d_0 = flip(tensor_3d, 0)
  expected_3d_0 = cv2.flip(data_3d, 0)
  assert_close(result_3d_0, expected_3d_0)

  # Flip around y-axis
  result_3d_1 = flip(tensor_3d, 1)
  expected_3d_1 = cv2.flip(data_3d, 1)
  assert_close(result_3d_1, expected_3d_1)

  # Flip around both axes
  result_3d_neg_1 = flip(tensor_3d, -1)
  expected_3d_neg_1 = cv2.flip(data_3d, -1)
  assert_close(result_3d_neg_1, expected_3d_neg_1)
