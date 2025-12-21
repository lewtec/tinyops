import numpy as np
from tinygrad import Tensor
from tinyops.stats.average import average
from tinyops._core import assert_close

def test_average_simple():
    a_np = np.arange(6).reshape(2, 3).astype(np.float32)
    a = Tensor(a_np)

    res = average(a)
    expected = np.average(a_np)
    assert_close(res, expected)

def test_average_axis():
    a_np = np.arange(6).reshape(2, 3).astype(np.float32)
    a = Tensor(a_np)

    res = average(a, axis=1)
    expected = np.average(a_np, axis=1)
    assert_close(res, expected)

def test_average_weights():
    a_np = np.arange(6).reshape(2, 3).astype(np.float32)
    w_np = np.random.rand(2, 3).astype(np.float32)
    a = Tensor(a_np)
    w = Tensor(w_np)

    res = average(a, weights=w)
    expected = np.average(a_np, weights=w_np)
    assert_close(res, expected)

def test_average_weights_axis_1d():
    a_np = np.arange(6).reshape(2, 3).astype(np.float32)
    w_np = np.array([0.1, 0.2, 0.7], dtype=np.float32)
    a = Tensor(a_np)
    w = Tensor(w_np)

    res = average(a, axis=1, weights=w)
    expected = np.average(a_np, axis=1, weights=w_np)
    assert_close(res, expected)

def test_average_returned():
    a_np = np.arange(6).reshape(2, 3).astype(np.float32)
    w_np = np.random.rand(2, 3).astype(np.float32)
    a = Tensor(a_np)
    w = Tensor(w_np)

    res, scl = average(a, weights=w, returned=True)
    exp_avg, exp_scl = np.average(a_np, weights=w_np, returned=True)

    assert_close(res, exp_avg)
    assert_close(scl, exp_scl)

def test_average_returned_no_weights():
    a_np = np.arange(6).reshape(2, 3).astype(np.float32)
    a = Tensor(a_np)

    res, scl = average(a, returned=True)
    exp_avg, exp_scl = np.average(a_np, returned=True)

    assert_close(res, exp_avg)
    assert_close(scl, exp_scl)
