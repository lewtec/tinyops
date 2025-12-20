import numpy as np
from tinygrad import Tensor
from tinyops.linalg.matmul import matmul
from tinyops._core import assert_close
import pytest

def test_matmul_2d():
    a = np.random.randn(3, 4).astype(np.float32)
    b = np.random.randn(4, 5).astype(np.float32)
    assert_close(matmul(Tensor(a), Tensor(b)), np.matmul(a, b))

def test_matmul_1d_1d():
    a = np.random.randn(3).astype(np.float32)
    b = np.random.randn(3).astype(np.float32)
    assert_close(matmul(Tensor(a), Tensor(b)), np.matmul(a, b))

def test_matmul_1d_2d():
    a = np.random.randn(3).astype(np.float32)
    b = np.random.randn(3, 4).astype(np.float32)
    assert_close(matmul(Tensor(a), Tensor(b)), np.matmul(a, b))

def test_matmul_2d_1d():
    a = np.random.randn(4, 3).astype(np.float32)
    b = np.random.randn(3).astype(np.float32)
    assert_close(matmul(Tensor(a), Tensor(b)), np.matmul(a, b))

def test_matmul_broadcast():
    a = np.random.randn(2, 3, 4).astype(np.float32)
    b = np.random.randn(4, 5).astype(np.float32)
    assert_close(matmul(Tensor(a), Tensor(b)), np.matmul(a, b))

    a = np.random.randn(3, 4).astype(np.float32)
    b = np.random.randn(2, 4, 5).astype(np.float32)
    assert_close(matmul(Tensor(a), Tensor(b)), np.matmul(a, b))

def test_matmul_broadcast_complex():
    a = np.random.randn(1, 3, 4).astype(np.float32)
    b = np.random.randn(2, 4, 5).astype(np.float32)
    assert_close(matmul(Tensor(a), Tensor(b)), np.matmul(a, b))

def test_matmul_scalar_raises():
    a = np.array(1.0).astype(np.float32)
    b = np.random.randn(3, 3).astype(np.float32)

    with pytest.raises(ValueError):
        matmul(Tensor(a), Tensor(b))

    with pytest.raises(ValueError):
        matmul(Tensor(b), Tensor(a))
