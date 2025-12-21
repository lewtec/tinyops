import numpy as np
from tinygrad import Tensor
from tinyops.stats.quantile import quantile
from tinyops._core import assert_close

def test_quantile_scalar():
    a_np = np.arange(12).reshape(3, 4).astype(np.float32)
    a = Tensor(a_np)

    res = quantile(a, 0.5)
    expected = np.quantile(a_np, 0.5)
    assert_close(res, expected)

def test_quantile_array():
    a_np = np.arange(12).reshape(3, 4).astype(np.float32)
    a = Tensor(a_np)

    q = [0.25, 0.75]
    res = quantile(a, q, axis=1)
    expected = np.quantile(a_np, q, axis=1)
    assert_close(res, expected)

def test_quantile_tensor_q():
    a_np = np.arange(12).reshape(3, 4).astype(np.float32)
    a = Tensor(a_np)

    q_np = np.array([0.25, 0.75], dtype=np.float32)
    q = Tensor(q_np)
    res = quantile(a, q, axis=1)
    expected = np.quantile(a_np, q_np, axis=1)
    assert_close(res, expected)
