import numpy as np
from tinygrad import Tensor
from tinyops.stats.std import std
from tinyops._core import assert_close

def test_std_default():
    a_np = np.random.randn(10).astype(np.float32)
    a = Tensor(a_np)

    res = std(a)
    expected = np.std(a_np) # ddof=0
    assert_close(res, expected)

def test_std_ddof1():
    a_np = np.random.randn(10).astype(np.float32)
    a = Tensor(a_np)

    res = std(a, ddof=1)
    expected = np.std(a_np, ddof=1)
    assert_close(res, expected)

def test_std_axis():
    a_np = np.random.randn(2, 3, 4).astype(np.float32)
    a = Tensor(a_np)

    res = std(a, axis=1)
    expected = np.std(a_np, axis=1)
    assert_close(res, expected)

def test_std_keepdims():
    a_np = np.random.randn(2, 3, 4).astype(np.float32)
    a = Tensor(a_np)

    res = std(a, axis=1, keepdims=True)
    expected = np.std(a_np, axis=1, keepdims=True)
    assert_close(res, expected)
