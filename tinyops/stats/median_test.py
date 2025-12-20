import numpy as np
from tinygrad import Tensor
from tinyops.stats.median import median
from tinyops._core import assert_close

def test_median_odd():
    a_np = np.array([1, 5, 2, 4, 3], dtype=np.float32)
    a = Tensor(a_np)

    res = median(a)
    expected = np.median(a_np)
    assert_close(res, expected)

def test_median_even():
    a_np = np.array([1, 5, 2, 4], dtype=np.float32)
    a = Tensor(a_np)

    res = median(a)
    expected = np.median(a_np)
    assert_close(res, expected)

def test_median_axis():
    a_np = np.random.randn(2, 3, 4).astype(np.float32)
    a = Tensor(a_np)

    res = median(a, axis=1)
    expected = np.median(a_np, axis=1)
    assert_close(res, expected)

def test_median_keepdims():
    a_np = np.random.randn(2, 3, 4).astype(np.float32)
    a = Tensor(a_np)

    res = median(a, axis=1, keepdims=True)
    expected = np.median(a_np, axis=1, keepdims=True)
    assert_close(res, expected)

def test_median_negative_axis():
    a_np = np.random.randn(2, 3, 4).astype(np.float32)
    a = Tensor(a_np)

    res = median(a, axis=-1)
    expected = np.median(a_np, axis=-1)
    assert_close(res, expected)
