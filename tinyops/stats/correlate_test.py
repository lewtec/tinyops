import numpy as np
from tinygrad import Tensor
from tinyops.stats.correlate import correlate
from tinyops.test_utils import assert_one_kernel
from tinyops._core import assert_close

@assert_one_kernel
def test_correlate_valid():
    a = np.array([1, 2, 3]).astype(np.float32)
    v = np.array([0, 1, 0.5]).astype(np.float32)
    result = correlate(Tensor(a), Tensor(v), mode="valid").realize()
    expected = np.correlate(a, v, mode="valid")
    assert_close(result, expected)

@assert_one_kernel
def test_correlate_same():
    a = np.array([1, 2, 3]).astype(np.float32)
    v = np.array([0, 1, 0.5]).astype(np.float32)
    result = correlate(Tensor(a), Tensor(v), mode="same").realize()
    expected = np.correlate(a, v, mode="same")
    assert_close(result, expected)

@assert_one_kernel
def test_correlate_full():
    a = np.array([1, 2, 3]).astype(np.float32)
    v = np.array([0, 1, 0.5]).astype(np.float32)
    result = correlate(Tensor(a), Tensor(v), mode="full").realize()
    expected = np.correlate(a, v, mode="full")
    assert_close(result, expected)
