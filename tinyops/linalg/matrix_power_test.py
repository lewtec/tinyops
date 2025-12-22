import numpy as np
import pytest
from tinygrad import Tensor
from tinyops.linalg.matrix_power import matrix_power
from tinyops.test_utils import assert_one_kernel
from tinyops._core import assert_close

@assert_one_kernel
def test_matrix_power_0():
    a_np = np.random.randn(3, 3).astype(np.float32)
    a = Tensor(a_np).realize()

    res = matrix_power(a, 0)
    expected = np.linalg.matrix_power(a_np, 0)
    assert_close(res, expected)

@assert_one_kernel
def test_matrix_power_1():
    a_np = np.random.randn(3, 3).astype(np.float32)
    a = Tensor(a_np).realize()

    res = matrix_power(a, 1)
    expected = np.linalg.matrix_power(a_np, 1)
    assert_close(res, expected)

@assert_one_kernel
def test_matrix_power_2():
    a_np = np.random.randn(3, 3).astype(np.float32)
    a = Tensor(a_np).realize()

    res = matrix_power(a, 2)
    expected = np.linalg.matrix_power(a_np, 2)
    assert_close(res, expected)

@assert_one_kernel
def test_matrix_power_3():
    a_np = np.random.randn(3, 3).astype(np.float32)
    a = Tensor(a_np).realize()

    res = matrix_power(a, 3)
    expected = np.linalg.matrix_power(a_np, 3)
    assert_close(res, expected)

@assert_one_kernel
def test_matrix_power_negative():
    # Use a well-conditioned matrix to avoid inverse issues
    # Ensure it's diagonally dominant
    a_np = np.random.randn(3, 3).astype(np.float32)
    a_np = a_np @ a_np.T + np.eye(3, dtype=np.float32)
    a = Tensor(a_np).realize()

    res = matrix_power(a, -1)
    expected = np.linalg.matrix_power(a_np, -1)
    # Inverse using Newton-Schulz has limits, use loose tolerance
    assert_close(res, expected, atol=1e-3, rtol=1e-3)

@assert_one_kernel
def test_matrix_power_batched():
    a_np = np.random.randn(2, 3, 3).astype(np.float32)
    a = Tensor(a_np).realize()

    res = matrix_power(a, 2)
    expected = np.linalg.matrix_power(a_np, 2)
    assert_close(res, expected)

@assert_one_kernel
def test_matrix_power_batched_0():
    a_np = np.random.randn(2, 3, 3).astype(np.float32)
    a = Tensor(a_np).realize()

    res = matrix_power(a, 0)
    expected = np.linalg.matrix_power(a_np, 0)
    assert_close(res, expected)
