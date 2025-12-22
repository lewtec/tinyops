import numpy as np
from tinygrad import Tensor
from tinyops.stats.std import std
from tinyops.test_utils import assert_one_kernel
from tinyops._core import assert_close

@assert_one_kernel
def test_std_default():
    a_np = np.random.randn(10).astype(np.float32)
    a = Tensor(a_np).realize()

    res = std(a).realize()
    expected = np.std(a_np) # ddof=0
    assert_close(res, expected)

@assert_one_kernel
def test_std_ddof1():
    a_np = np.random.randn(10).astype(np.float32)
    a = Tensor(a_np).realize()

    res = std(a, ddof=1).realize()
    expected = np.std(a_np, ddof=1)
    assert_close(res, expected)

@assert_one_kernel
def test_std_axis():
    a_np = np.random.randn(2, 3, 4).astype(np.float32)
    a = Tensor(a_np).realize()

    res = std(a, axis=1).realize()
    expected = np.std(a_np, axis=1)
    assert_close(res, expected)

@assert_one_kernel
def test_std_keepdims():
    a_np = np.random.randn(2, 3, 4).astype(np.float32)
    a = Tensor(a_np).realize()

    res = std(a, axis=1, keepdims=True).realize()
    expected = np.std(a_np, axis=1, keepdims=True)
    assert_close(res, expected)
