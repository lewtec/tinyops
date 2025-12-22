import numpy as np
from tinygrad import Tensor
from tinyops.linalg.trace import trace
from tinyops.test_utils import assert_one_kernel
from tinyops._core import assert_close
import pytest

@assert_one_kernel
def test_trace_basic():
    a_np = np.arange(9).reshape(3, 3).astype(np.float32)
    a = Tensor(a_np).realize()
    assert_close(trace(a), np.trace(a_np))

@assert_one_kernel
def test_trace_offset():
    a_np = np.arange(12).reshape(4, 3).astype(np.float32)
    a = Tensor(a_np).realize()
    assert_close(trace(a, offset=1), np.trace(a_np, offset=1)).realize()
    assert_close(trace(a, offset=-1), np.trace(a_np, offset=-1)).realize()
    assert_close(trace(a, offset=2), np.trace(a_np, offset=2)).realize()

@assert_one_kernel
def test_trace_axes():
    a_np = np.arange(24).reshape(2, 3, 4).astype(np.float32)
    a = Tensor(a_np).realize()
    # Trace over last two axes (default is 0, 1)
    # Default np.trace uses 0, 1.
    assert_close(trace(a), np.trace(a_np))

    # Trace over 1, 2
    assert_close(trace(a, axis1=1, axis2=2), np.trace(a_np, axis1=1, axis2=2)).realize()

@assert_one_kernel
def test_trace_empty():
    a_np = np.zeros((3, 3), dtype=np.float32)
    a = Tensor(a_np).realize()
    assert_close(trace(a, offset=10), np.trace(a_np, offset=10)).realize()
