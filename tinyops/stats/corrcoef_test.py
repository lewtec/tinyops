import numpy as np
from tinygrad import Tensor

from tinyops._core import assert_close, assert_one_kernel
from tinyops.stats.corrcoef import corrcoef


@assert_one_kernel
def test_corrcoef_basic():
    data = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
    result = corrcoef(Tensor(data))
    expected = np.corrcoef(data)
    assert_close(result, expected)


@assert_one_kernel
def test_corrcoef_y():
    x = np.array([1, 2, 3]).astype(np.float32)
    y = np.array([4, 5, 6]).astype(np.float32)
    result = corrcoef(Tensor(x), Tensor(y))
    expected = np.corrcoef(x, y)
    assert_close(result, expected)


@assert_one_kernel
def test_corrcoef_rowvar_false():
    data = np.array([[1, 4], [2, 5], [3, 6]]).astype(np.float32)
    result = corrcoef(Tensor(data), rowvar=False)
    expected = np.corrcoef(data, rowvar=False)
    assert_close(result, expected)
