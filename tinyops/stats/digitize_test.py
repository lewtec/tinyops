import numpy as np
from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.test_utils import assert_one_kernel
from tinyops.stats.digitize import digitize

@assert_one_kernel
def test_digitize_simple():
  x = np.array([0.2, 6.4, 3.0, 1.6])
  bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])

  result = digitize(Tensor(x), Tensor(bins))
  expected = np.digitize(x, bins)

  assert_close(result, expected)

@assert_one_kernel
def test_digitize_right():
  x = np.array([0.2, 6.4, 3.0, 1.6])
  bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])

  result = digitize(Tensor(x), Tensor(bins), right=True)
  expected = np.digitize(x, bins, right=True)

  assert_close(result, expected)

@assert_one_kernel
def test_digitize_2d():
  x = np.array([[0.2, 6.4], [3.0, 1.6]])
  bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])

  result = digitize(Tensor(x), Tensor(bins))
  expected = np.digitize(x, bins)

  assert_close(result, expected)

@assert_one_kernel
def test_digitize_edge_cases():
  x = np.array([-1.0, 0.0, 1.0, 10.0, 11.0])
  bins = np.array([0.0, 1.0, 10.0])

  # Test with right=False
  result_left = digitize(Tensor(x), Tensor(bins))
  expected_left = np.digitize(x, bins)
  assert_close(result_left, expected_left)

  # Test with right=True
  result_right = digitize(Tensor(x), Tensor(bins), right=True)
  expected_right = np.digitize(x, bins, right=True)
  assert_close(result_right, expected_right)

@assert_one_kernel
def test_digitize_empty_input():
    x = np.array([])
    bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])

    result = digitize(Tensor(x), Tensor(bins))
    expected = np.digitize(x, bins)

    assert result.shape == expected.shape

@assert_one_kernel
def test_digitize_empty_bins():
    x = np.array([1, 2, 3])
    bins = np.array([])

    result = digitize(Tensor(x), Tensor(bins))
    expected = np.digitize(x, bins)

    assert_close(result, expected)
