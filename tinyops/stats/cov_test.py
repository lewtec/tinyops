import numpy as np
import pytest
from tinygrad import Tensor, dtypes
from tinyops.stats.cov import cov
from tinyops.test_utils import assert_one_kernel
from tinyops._core import assert_close

@assert_one_kernel
def test_cov_basic():
    data = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
    result = cov(Tensor(data)).realize()
    expected = np.cov(data)
    assert_close(result, expected)

@assert_one_kernel
def test_cov_y():
    x = np.array([1, 2, 3]).astype(np.float32)
    y = np.array([4, 5, 6]).astype(np.float32)
    result = cov(Tensor(x), Tensor(y)).realize()
    expected = np.cov(x, y)
    assert_close(result, expected)

@assert_one_kernel
def test_cov_rowvar_false():
    data = np.array([[1, 4], [2, 5], [3, 6]]).astype(np.float32)
    result = cov(Tensor(data), rowvar=False).realize()
    expected = np.cov(data, rowvar=False)
    assert_close(result, expected)

@assert_one_kernel
def test_cov_ddof():
    data = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
    result = cov(Tensor(data), ddof=0).realize()
    expected = np.cov(data, ddof=0)
    assert_close(result, expected)

def test_cov_empty():
    data = Tensor([], dtype=dtypes.float32).reshape(0, 2)
    result = cov(data)
    assert result.shape == (0,)

def test_cov_ndim():
    data = Tensor(np.zeros((2, 3, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        cov(data)
