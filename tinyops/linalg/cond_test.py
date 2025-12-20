import numpy as np
from tinygrad import Tensor
from tinyops.linalg.cond import cond
from tinyops._core import assert_close
import pytest

def test_cond_basic():
    np.random.seed(42)
    a_np = np.random.randn(5, 5).astype(np.float32) + np.eye(5) * 5
    a = Tensor(a_np)

    assert_close(cond(a, p='fro'), np.linalg.cond(a_np, p='fro'))
    assert_close(cond(a, p=1), np.linalg.cond(a_np, p=1))
    assert_close(cond(a, p=np.inf), np.linalg.cond(a_np, p=np.inf))
