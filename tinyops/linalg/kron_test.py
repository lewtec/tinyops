import numpy as np
from tinygrad import Tensor
from tinyops.linalg.kron import kron
from tinyops._core import assert_close

def test_kron_1d():
    a_np = np.array([1, 10, 100], dtype=np.float32)
    b_np = np.array([5, 6], dtype=np.float32)
    a = Tensor(a_np)
    b = Tensor(b_np)

    res = kron(a, b)
    expected = np.kron(a_np, b_np)
    assert_close(res, expected)

def test_kron_2d():
    a_np = np.eye(2, dtype=np.float32)
    b_np = np.ones((2, 2), dtype=np.float32)
    a = Tensor(a_np)
    b = Tensor(b_np)

    res = kron(a, b)
    expected = np.kron(a_np, b_np)
    assert_close(res, expected)

def test_kron_diff_dims():
    a_np = np.eye(2, dtype=np.float32)
    b_np = np.array([1, 2], dtype=np.float32)
    a = Tensor(a_np)
    b = Tensor(b_np)

    res = kron(a, b)
    expected = np.kron(a_np, b_np)
    assert_close(res, expected)

def test_kron_diff_dims_2():
    a_np = np.array([1, 2], dtype=np.float32)
    b_np = np.eye(2, dtype=np.float32)
    a = Tensor(a_np)
    b = Tensor(b_np)

    res = kron(a, b)
    expected = np.kron(a_np, b_np)
    assert_close(res, expected)
