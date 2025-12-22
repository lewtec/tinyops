import numpy as np
from tinygrad import Tensor
from tinyops.stats.mean import mean
from tinyops.test_utils import assert_one_kernel
from tinyops._core import assert_close

@assert_one_kernel
def test_mean_all():
    a_np = np.random.randn(2, 3, 4).astype(np.float32)
    a = Tensor(a_np).realize()

    res = mean(a).realize()
    expected = np.mean(a_np)
    assert_close(res, expected)

@assert_one_kernel
def test_mean_axis():
    a_np = np.random.randn(2, 3, 4).astype(np.float32)
    a = Tensor(a_np).realize()

    res = mean(a, axis=1).realize()
    expected = np.mean(a_np, axis=1)
    assert_close(res, expected)

@assert_one_kernel
def test_mean_keepdims():
    a_np = np.random.randn(2, 3, 4).astype(np.float32)
    a = Tensor(a_np).realize()

    res = mean(a, axis=1, keepdims=True).realize()
    expected = np.mean(a_np, axis=1, keepdims=True)
    assert_close(res, expected)

@assert_one_kernel
def test_mean_tuple_axis():
    a_np = np.random.randn(2, 3, 4).astype(np.float32)
    a = Tensor(a_np).realize()

    res = mean(a, axis=(0, 2)).realize()
    expected = np.mean(a_np, axis=(0, 2)).realize()
    assert_close(res, expected)
