import numpy as np
from tinygrad import Tensor
from tinyops.stats.bincount import bincount
from tinyops.test_utils import assert_one_kernel
from tinyops._core import assert_close
import pytest

@assert_one_kernel
def test_bincount_basic():
    x_np = np.array([0, 1, 1, 3, 2, 1, 7])
    x = Tensor(x_np).realize()
    res = bincount(x).realize()
    expected = np.bincount(x_np)
    assert_close(res, expected)

@assert_one_kernel
def test_bincount_weights():
    x_np = np.array([0, 1, 1, 3, 2, 1, 7])
    w_np = np.array([0.3, 0.5, 0.2, 0.7, 1.0, -0.6, 2.0])
    x = Tensor(x_np).realize()
    w = Tensor(w_np).realize()
    res = bincount(x, weights=w).realize()
    expected = np.bincount(x_np, weights=w_np)
    assert_close(res, expected)

@assert_one_kernel
def test_bincount_minlength():
    x_np = np.array([0, 1, 1, 3])
    x = Tensor(x_np).realize()
    res = bincount(x, minlength=10).realize()
    expected = np.bincount(x_np, minlength=10)
    assert_close(res, expected)

@assert_one_kernel
def test_bincount_empty():
    x_np = np.array([], dtype=int)
    x = Tensor(x_np).realize()
    res = bincount(x).realize()
    expected = np.bincount(x_np)
    assert_close(res, expected)

@assert_one_kernel
def test_bincount_empty_minlength():
    x_np = np.array([], dtype=int)
    x = Tensor(x_np).realize()
    res = bincount(x, minlength=5).realize()
    expected = np.bincount(x_np, minlength=5)
    assert_close(res, expected)

@assert_one_kernel
def test_bincount_negative_raises():
    x = Tensor([-1, 0]).realize()
    with pytest.raises(ValueError):
        bincount(x)
