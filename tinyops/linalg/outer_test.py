import numpy as np
from tinygrad import Tensor

from tinyops._core import assert_close, assert_one_kernel
from tinyops.linalg.outer import outer


@assert_one_kernel
def test_outer_1d():
    a = np.random.randn(3).astype(np.float32)
    b = np.random.randn(4).astype(np.float32)
    assert_close(outer(Tensor(a), Tensor(b)), np.outer(a, b))


@assert_one_kernel
def test_outer_nd():
    a = np.random.randn(2, 3).astype(np.float32)
    b = np.random.randn(2, 2).astype(np.float32)
    assert_close(outer(Tensor(a), Tensor(b)), np.outer(a, b))


@assert_one_kernel
def test_outer_scalar():
    a = np.array(2.0).astype(np.float32)
    b = np.random.randn(3).astype(np.float32)
    assert_close(outer(Tensor(a), Tensor(b)), np.outer(a, b))
