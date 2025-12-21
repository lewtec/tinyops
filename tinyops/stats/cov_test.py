import numpy as np
from tinygrad import Tensor
from tinyops.stats.cov import cov
from tinyops._core import assert_close

def test_cov_basic():
    data = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
    result = cov(Tensor(data))
    expected = np.cov(data)
    assert_close(result, expected)

def test_cov_y():
    x = np.array([1, 2, 3]).astype(np.float32)
    y = np.array([4, 5, 6]).astype(np.float32)
    result = cov(Tensor(x), Tensor(y))
    expected = np.cov(x, y)
    assert_close(result, expected)

def test_cov_rowvar_false():
    data = np.array([[1, 4], [2, 5], [3, 6]]).astype(np.float32)
    result = cov(Tensor(data), rowvar=False)
    expected = np.cov(data, rowvar=False)
    assert_close(result, expected)

def test_cov_ddof():
    data = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
    result = cov(Tensor(data), ddof=0)
    expected = np.cov(data, ddof=0)
    assert_close(result, expected)
