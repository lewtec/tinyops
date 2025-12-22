import numpy as np
from tinygrad import Tensor
from tinyops.stats.ptp import ptp
from tinyops.test_utils import assert_one_kernel
from tinyops._core import assert_close

@assert_one_kernel
def test_ptp_all():
    a_np = np.random.randn(2, 3, 4).astype(np.float32)
    a = Tensor(a_np).realize()
    res = ptp(a)
    expected = np.ptp(a_np)
    assert_close(res, expected)

@assert_one_kernel
def test_ptp_axis():
    a_np = np.random.randn(2, 3, 4).astype(np.float32)
    a = Tensor(a_np).realize()
    res = ptp(a, axis=1)
    expected = np.ptp(a_np, axis=1)
    assert_close(res, expected)

@assert_one_kernel
def test_ptp_keepdims():
    a_np = np.random.randn(2, 3, 4).astype(np.float32)
    a = Tensor(a_np).realize()
    res = ptp(a, axis=1, keepdims=True)
    expected = np.ptp(a_np, axis=1, keepdims=True)
    assert_close(res, expected)

@assert_one_kernel
def test_ptp_tuple_axis():
    a_np = np.random.randn(2, 3, 4).astype(np.float32)
    a = Tensor(a_np).realize()
    res = ptp(a, axis=(0, 2))
    expected = np.ptp(a_np, axis=(0, 2))
    assert_close(res, expected)
