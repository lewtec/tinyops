import numpy as np
from tinygrad import Tensor

from tinyops._core import assert_close, assert_one_kernel
from tinyops.linalg.inner import inner


@assert_one_kernel
def test_inner_1d():
    a = np.random.randn(3).astype(np.float32)
    b = np.random.randn(3).astype(np.float32)
    assert_close(inner(Tensor(a), Tensor(b)), np.inner(a, b))


@assert_one_kernel
def test_inner_2d():
    a = np.random.randn(2, 3).astype(np.float32)
    b = np.random.randn(4, 3).astype(np.float32)
    assert_close(inner(Tensor(a), Tensor(b)), np.inner(a, b))


@assert_one_kernel
def test_inner_nd():
    a = np.random.randn(2, 3, 4).astype(np.float32)
    b = np.random.randn(5, 4).astype(np.float32)
    assert_close(inner(Tensor(a), Tensor(b)), np.inner(a, b))


@assert_one_kernel
def test_inner_scalar():
    a = np.array(2.0).astype(np.float32)
    b = np.random.randn(3).astype(np.float32)
    assert_close(inner(Tensor(a), Tensor(b)), np.inner(a, b))

    a = np.random.randn(2, 3).astype(np.float32)
    b = np.array(3.0).astype(np.float32)
    assert_close(inner(Tensor(a), Tensor(b)), np.inner(a, b))
