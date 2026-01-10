import numpy as np
from tinygrad import Tensor
from tinyops.linalg.vdot import vdot
from tinyops._core import assert_one_kernel
from tinyops._core import assert_close
import pytest

@assert_one_kernel
def test_vdot_1d():
    a = np.random.randn(3).astype(np.float32)
    b = np.random.randn(3).astype(np.float32)
    assert_close(vdot(Tensor(a), Tensor(b)), np.vdot(a, b))

@assert_one_kernel
def test_vdot_md():
    a = np.random.randn(2, 3).astype(np.float32)
    b = np.random.randn(2, 3).astype(np.float32)
    assert_close(vdot(Tensor(a), Tensor(b)), np.vdot(a, b))

@assert_one_kernel
def test_vdot_flatten_mismatch():
    a = np.random.randn(2, 3).astype(np.float32) # size 6
    b = np.random.randn(6).astype(np.float32) # size 6
    assert_close(vdot(Tensor(a), Tensor(b)), np.vdot(a, b))

@assert_one_kernel
def test_vdot_size_mismatch_raises():
    a = np.random.randn(2).astype(np.float32)
    b = np.random.randn(3).astype(np.float32)

    # tinygrad raises ValueError/RuntimeError on shape mismatch
    # np raises ValueError
    with pytest.raises(Exception):
        vdot(Tensor(a), Tensor(b))
