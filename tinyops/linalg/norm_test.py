import numpy as np
from tinygrad import Tensor

from tinyops._core import assert_close, assert_one_kernel
from tinyops.linalg.norm import norm


@assert_one_kernel
def test_norm_vector():
    a_np = np.arange(9).astype(np.float32) - 4
    a = Tensor(a_np).realize()

    assert_close(norm(a), np.linalg.norm(a_np))
    assert_close(norm(a, ord=1), np.linalg.norm(a_np, ord=1))
    assert_close(norm(a, ord=np.inf), np.linalg.norm(a_np, ord=np.inf))
    assert_close(norm(a, ord=-np.inf), np.linalg.norm(a_np, ord=-np.inf))
    assert_close(norm(a, ord=3), np.linalg.norm(a_np, ord=3))


@assert_one_kernel
def test_norm_matrix():
    a_np = np.arange(9).reshape(3, 3).astype(np.float32) - 4
    a = Tensor(a_np).realize()

    assert_close(norm(a), np.linalg.norm(a_np))  # fro
    assert_close(norm(a, ord="fro"), np.linalg.norm(a_np, ord="fro"))
    assert_close(norm(a, ord=1), np.linalg.norm(a_np, ord=1))
    assert_close(norm(a, ord=-1), np.linalg.norm(a_np, ord=-1))
    assert_close(norm(a, ord=np.inf), np.linalg.norm(a_np, ord=np.inf))
    assert_close(norm(a, ord=-np.inf), np.linalg.norm(a_np, ord=-np.inf))


@assert_one_kernel
def test_norm_axis():
    a_np = np.arange(24).reshape(2, 3, 4).astype(np.float32)
    a = Tensor(a_np).realize()

    assert_close(norm(a, axis=1), np.linalg.norm(a_np, axis=1))
    assert_close(norm(a, axis=(1, 2)), np.linalg.norm(a_np, axis=(1, 2)))
