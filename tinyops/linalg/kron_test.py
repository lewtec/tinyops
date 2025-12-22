import numpy as np
from tinygrad import Tensor
from tinyops.linalg.kron import kron
from tinyops.test_utils import assert_one_kernel
from tinyops._core import assert_close

@assert_one_kernel
def test_kron_1d():
    a_np = np.array([1, 10, 100], dtype=np.float32)
    b_np = np.array([5, 6], dtype=np.float32)
    a = Tensor(a_np).realize()
    b = Tensor(b_np).realize()

    res = kron(a, b).realize()
    expected = np.kron(a_np, b_np)
    assert_close(res, expected)

@assert_one_kernel
def test_kron_2d():
    a_np = np.eye(2, dtype=np.float32)
    b_np = np.ones((2, 2), dtype=np.float32)
    a = Tensor(a_np).realize()
    b = Tensor(b_np).realize()

    res = kron(a, b).realize()
    expected = np.kron(a_np, b_np)
    assert_close(res, expected)

@assert_one_kernel
def test_kron_diff_dims():
    a_np = np.eye(2, dtype=np.float32)
    b_np = np.array([1, 2], dtype=np.float32)
    a = Tensor(a_np).realize()
    b = Tensor(b_np).realize()

    res = kron(a, b).realize()
    expected = np.kron(a_np, b_np)
    assert_close(res, expected)

@assert_one_kernel
def test_kron_diff_dims_2():
    a_np = np.array([1, 2], dtype=np.float32)
    b_np = np.eye(2, dtype=np.float32)
    a = Tensor(a_np).realize()
    b = Tensor(b_np).realize()

    res = kron(a, b).realize()
    expected = np.kron(a_np, b_np)
    assert_close(res, expected)
