import numpy as np
import pytest
from tinygrad import Tensor

from tinyops._core import assert_close, assert_one_kernel
from tinyops.linalg.dot import dot


@assert_one_kernel
def test_dot_1d():
    a = np.random.randn(3).astype(np.float32)
    b = np.random.randn(3).astype(np.float32)

    assert_close(dot(Tensor(a), Tensor(b)), np.dot(a, b))


@assert_one_kernel
def test_dot_2d():
    a = np.random.randn(3, 4).astype(np.float32)
    b = np.random.randn(4, 5).astype(np.float32)

    assert_close(dot(Tensor(a), Tensor(b)), np.dot(a, b))


@assert_one_kernel
def test_dot_scalar():
    a = np.array(np.random.randn()).astype(np.float32)
    b = np.random.randn(3, 4).astype(np.float32)

    # 0D, 2D
    assert_close(dot(Tensor(a), Tensor(b)), np.dot(a, b))

    # 2D, 0D
    assert_close(dot(Tensor(b), Tensor(a)), np.dot(b, a))

    # 0D, 0D
    assert_close(dot(Tensor(a), Tensor(a)), np.dot(a, a))


@assert_one_kernel
def test_dot_nd_1d():
    a = np.random.randn(2, 3, 4).astype(np.float32)
    b = np.random.randn(4).astype(np.float32)

    assert_close(dot(Tensor(a), Tensor(b)), np.dot(a, b))


@assert_one_kernel
def test_dot_nd_nd():
    # a: (2, 3, 4), b: (5, 4, 6) -> (2, 3, 5, 6)
    a = np.random.randn(2, 3, 4).astype(np.float32)
    b = np.random.randn(5, 4, 6).astype(np.float32)

    assert_close(dot(Tensor(a), Tensor(b)), np.dot(a, b))


@assert_one_kernel
def test_dot_mismatch():
    a = np.random.randn(3, 4).astype(np.float32)
    b = np.random.randn(3, 5).astype(np.float32)

    with pytest.raises(ValueError):
        dot(Tensor(a), Tensor(b))
