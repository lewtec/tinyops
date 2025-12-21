import numpy as np
from tinygrad import Tensor
from tinyops.stats.percentile import percentile
from tinyops._core import assert_close

def test_percentile_scalar():
    a_np = np.arange(12).reshape(3, 4).astype(np.float32)
    a = Tensor(a_np)

    res = percentile(a, 50)
    expected = np.percentile(a_np, 50)
    assert_close(res, expected)

def test_percentile_scalar_axis():
    a_np = np.arange(12).reshape(3, 4).astype(np.float32)
    a = Tensor(a_np)

    res = percentile(a, 50, axis=1)
    expected = np.percentile(a_np, 50, axis=1)
    assert_close(res, expected)

def test_percentile_array():
    a_np = np.arange(12).reshape(3, 4).astype(np.float32)
    a = Tensor(a_np)

    q = [25, 75]
    res = percentile(a, q, axis=1)
    expected = np.percentile(a_np, q, axis=1)
    assert_close(res, expected)

def test_percentile_keepdims():
    a_np = np.arange(12).reshape(3, 4).astype(np.float32)
    a = Tensor(a_np)

    res = percentile(a, 50, axis=1, keepdims=True)
    expected = np.percentile(a_np, 50, axis=1, keepdims=True)
    assert_close(res, expected)

def test_percentile_array_keepdims():
    a_np = np.arange(12).reshape(3, 4).astype(np.float32)
    a = Tensor(a_np)

    q = [25, 75]
    res = percentile(a, q, axis=1, keepdims=True)
    expected = np.percentile(a_np, q, axis=1, keepdims=True)
    assert_close(res, expected)
