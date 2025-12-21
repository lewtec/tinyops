import cv2
import numpy as np
from tinygrad import Tensor
from tinyops.image.rotate import rotate, ROTATE_90_CLOCKWISE, ROTATE_180, ROTATE_90_COUNTERCLOCKWISE
from tinyops._core import assert_close

def test_rotate():
  # Test with a 2D array
  data = np.random.randn(10, 20).astype(np.float32)
  tensor = Tensor(data)

  # Rotate 90 degrees clockwise
  result_0 = rotate(tensor, ROTATE_90_CLOCKWISE)
  expected_0 = cv2.rotate(data, cv2.ROTATE_90_CLOCKWISE)
  assert_close(result_0, expected_0)

  # Rotate 180 degrees
  result_1 = rotate(tensor, ROTATE_180)
  expected_1 = cv2.rotate(data, cv2.ROTATE_180)
  assert_close(result_1, expected_1)

  # Rotate 90 degrees counter-clockwise
  result_2 = rotate(tensor, ROTATE_90_COUNTERCLOCKWISE)
  expected_2 = cv2.rotate(data, cv2.ROTATE_90_COUNTERCLOCKWISE)
  assert_close(result_2, expected_2)

  # Test with a 3D array (image)
  data_3d = np.random.randn(10, 20, 3).astype(np.float32)
  tensor_3d = Tensor(data_3d)

  # Rotate 90 degrees clockwise
  result_3d_0 = rotate(tensor_3d, ROTATE_90_CLOCKWISE)
  expected_3d_0 = cv2.rotate(data_3d, cv2.ROTATE_90_CLOCKWISE)
  assert_close(result_3d_0, expected_3d_0)

  # Rotate 180 degrees
  result_3d_1 = rotate(tensor_3d, ROTATE_180)
  expected_3d_1 = cv2.rotate(data_3d, cv2.ROTATE_180)
  assert_close(result_3d_1, expected_3d_1)

  # Rotate 90 degrees counter-clockwise
  result_3d_2 = rotate(tensor_3d, ROTATE_90_COUNTERCLOCKWISE)
  expected_3d_2 = cv2.rotate(data_3d, cv2.ROTATE_90_COUNTERCLOCKWISE)
  assert_close(result_3d_2, expected_3d_2)
