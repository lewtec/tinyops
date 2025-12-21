import cv2
import numpy as np
from tinygrad import Tensor
from tinyops.image.resize import resize, INTER_NEAREST, INTER_LINEAR
from tinyops._core import assert_close

def test_resize_nearest():
  # Test with a 2D array
  data = np.random.randn(10, 20).astype(np.float32)
  tensor = Tensor(data)

  dsize = (5, 10)
  result = resize(tensor, dsize, interpolation=INTER_NEAREST)
  expected = cv2.resize(data, (dsize[1], dsize[0]), interpolation=cv2.INTER_NEAREST)
  assert_close(result, expected, atol=1e-5, rtol=1e-5)

  # Test with a 3D array (image)
  data_3d = np.random.randn(10, 20, 3).astype(np.float32)
  tensor_3d = Tensor(data_3d)

  dsize_3d = (15, 25)
  result_3d = resize(tensor_3d, dsize_3d, interpolation=INTER_NEAREST)
  expected_3d = cv2.resize(data_3d, (dsize_3d[1], dsize_3d[0]), interpolation=cv2.INTER_NEAREST)
  assert_close(result_3d, expected_3d, atol=1e-5, rtol=1e-5)

def test_resize_linear():
  # Test with a 2D array
  data = np.random.randn(10, 20).astype(np.float32)
  tensor = Tensor(data)

  dsize = (5, 10)
  result = resize(tensor, dsize, interpolation=INTER_LINEAR)
  expected = cv2.resize(data, (dsize[1], dsize[0]), interpolation=cv2.INTER_LINEAR)
  assert_close(result, expected, atol=1e-5, rtol=1e-5)

  # Test with a 3D array (image)
  data_3d = np.random.randn(10, 20, 3).astype(np.float32)
  tensor_3d = Tensor(data_3d)

  dsize_3d = (15, 25)
  result_3d = resize(tensor_3d, dsize_3d, interpolation=INTER_LINEAR)
  expected_3d = cv2.resize(data_3d, (dsize_3d[1], dsize_3d[0]), interpolation=cv2.INTER_LINEAR)
  assert_close(result_3d, expected_3d, atol=1e-5, rtol=1e-5)
