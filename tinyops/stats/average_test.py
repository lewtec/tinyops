import numpy as np
from tinygrad import Tensor
from tinyops.stats.average import average
from tinyops.test_utils import assert_one_kernel
from tinyops._core import assert_close


@assert_one_kernel
def test_average_simple():
    a_np = np.arange(6).reshape(2, 3).astype(np.float32)
    a = Tensor(a_np).realize()

    res = average(a).realize()
    expected = np.average(a_np)
    assert_close(res, expected)


@assert_one_kernel
def test_average_axis():
    a_np = np.arange(6).reshape(2, 3).astype(np.float32)
    a = Tensor(a_np).realize()

    res = average(a, axis=1).realize()
    expected = np.average(a_np, axis=1)
    assert_close(res, expected)


@assert_one_kernel
def test_average_weights():
    a_np = np.arange(6).reshape(2, 3).astype(np.float32)
    w_np = np.random.rand(2, 3).astype(np.float32)
    a = Tensor(a_np).realize()
    w = Tensor(w_np).realize()

    res = average(a, weights=w).realize()
    expected = np.average(a_np, weights=w_np)
    assert_close(res, expected)


@assert_one_kernel
def test_average_weights_axis_1d():
    a_np = np.arange(6).reshape(2, 3).astype(np.float32)
    w_np = np.array([0.1, 0.2, 0.7], dtype=np.float32)
    a = Tensor(a_np).realize()
    w = Tensor(w_np).realize()

    res = average(a, axis=1, weights=w).realize()
    expected = np.average(a_np, axis=1, weights=w_np)
    assert_close(res, expected)


@assert_one_kernel
def test_average_returned():
    a_np = np.arange(6).reshape(2, 3).astype(np.float32)
    w_np = np.random.rand(2, 3).astype(np.float32)
    a = Tensor(a_np).realize()
    w = Tensor(w_np).realize()

    res, scl = average(a, weights=w, returned=True)
    res.realize()
    scl.realize()
    exp_avg, exp_scl = np.average(a_np, weights=w_np, returned=True)

    assert_close(res, exp_avg)
    assert_close(scl, exp_scl)


@assert_one_kernel
def test_average_returned_no_weights():
    a_np = np.arange(6).reshape(2, 3).astype(np.float32)
    a = Tensor(a_np).realize()

    res, scl = average(a, returned=True)
    res.realize()
    scl.realize()
    exp_avg, exp_scl = np.average(a_np, returned=True)

    assert_close(res, exp_avg)
    assert_close(scl, exp_scl)
