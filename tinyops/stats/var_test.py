import numpy as np
from tinygrad import Tensor
from tinyops.stats.var import var
from tinyops._core import assert_close

def test_var_default():
    a_np = np.random.randn(10).astype(np.float32)
    a = Tensor(a_np)

    res = var(a)
    expected = np.var(a_np) # ddof=0
    assert_close(res, expected)

def test_var_ddof1():
    a_np = np.random.randn(10).astype(np.float32)
    a = Tensor(a_np)

    res = var(a, ddof=1)
    expected = np.var(a_np, ddof=1)
    assert_close(res, expected)

def test_var_axis():
    a_np = np.random.randn(2, 3, 4).astype(np.float32)
    a = Tensor(a_np)

    res = var(a, axis=1)
    expected = np.var(a_np, axis=1)
    assert_close(res, expected)

def test_var_keepdims():
    a_np = np.random.randn(2, 3, 4).astype(np.float32)
    a = Tensor(a_np)

    res = var(a, axis=1, keepdims=True)
    expected = np.var(a_np, axis=1, keepdims=True)
    assert_close(res, expected)
