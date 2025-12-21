import numpy as np
from tinygrad import Tensor
from tinyops.stats.bincount import bincount
from tinyops._core import assert_close
import pytest

def test_bincount_basic():
    x_np = np.array([0, 1, 1, 3, 2, 1, 7])
    x = Tensor(x_np)
    res = bincount(x)
    expected = np.bincount(x_np)
    assert_close(res, expected)

def test_bincount_weights():
    x_np = np.array([0, 1, 1, 3, 2, 1, 7])
    w_np = np.array([0.3, 0.5, 0.2, 0.7, 1.0, -0.6, 2.0])
    x = Tensor(x_np)
    w = Tensor(w_np)
    res = bincount(x, weights=w)
    expected = np.bincount(x_np, weights=w_np)
    assert_close(res, expected)

def test_bincount_minlength():
    x_np = np.array([0, 1, 1, 3])
    x = Tensor(x_np)
    res = bincount(x, minlength=10)
    expected = np.bincount(x_np, minlength=10)
    assert_close(res, expected)

def test_bincount_empty():
    x_np = np.array([], dtype=int)
    x = Tensor(x_np)
    res = bincount(x)
    expected = np.bincount(x_np)
    assert_close(res, expected)

def test_bincount_empty_minlength():
    x_np = np.array([], dtype=int)
    x = Tensor(x_np)
    res = bincount(x, minlength=5)
    expected = np.bincount(x_np, minlength=5)
    assert_close(res, expected)

def test_bincount_negative_raises():
    x = Tensor([-1, 0])
    with pytest.raises(ValueError):
        bincount(x)
