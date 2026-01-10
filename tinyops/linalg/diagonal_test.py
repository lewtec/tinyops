import numpy as np
from tinygrad import Tensor
from tinyops.linalg.diagonal import diagonal
from tinyops._core import assert_one_kernel
from tinyops._core import assert_close

@assert_one_kernel
def test_diagonal_2d():
    a_np = np.arange(12).reshape(3, 4).astype(np.float32)
    a = Tensor(a_np).realize()

    # offset 0
    res = diagonal(a, 0)
    expected = np.diagonal(a_np, 0)
    assert_close(res, expected)

    # offset 1
    res = diagonal(a, 1)
    expected = np.diagonal(a_np, 1)
    assert_close(res, expected)

    # offset -1
    res = diagonal(a, -1)
    expected = np.diagonal(a_np, -1)
    assert_close(res, expected)

@assert_one_kernel
def test_diagonal_3d():
    a_np = np.arange(24).reshape(2, 3, 4).astype(np.float32)
    a = Tensor(a_np).realize()

    # axis1=0, axis2=1 -> (4, 2)
    res = diagonal(a, 0, 0, 1)
    expected = np.diagonal(a_np, 0, 0, 1)
    assert_close(res, expected)

    # axis1=1, axis2=2 -> (2, 3)
    res = diagonal(a, 0, 1, 2)
    expected = np.diagonal(a_np, 0, 1, 2)
    assert_close(res, expected)

@assert_one_kernel
def test_diagonal_large_offset():
    a_np = np.eye(3, dtype=np.float32)
    a = Tensor(a_np).realize()

    res = diagonal(a, 10)
    expected = np.diagonal(a_np, 10)
    # This might return empty tensor
    # assert_close might fail if shapes mismatch or if numpy returns (0,) and we return (0,)
    assert_close(res, expected)

@assert_one_kernel
def test_diagonal_negative_axes():
    a_np = np.arange(24).reshape(2, 3, 4).astype(np.float32)
    a = Tensor(a_np).realize()

    res = diagonal(a, 0, -2, -1)
    expected = np.diagonal(a_np, 0, -2, -1)
    assert_close(res, expected)
