import numpy as np
from tinygrad import Tensor
from tinyops.linalg.det import det
from tinyops.test_utils import assert_one_kernel
from tinyops._core import assert_close

@assert_one_kernel
def test_det_identity():
    a_np = np.eye(3, dtype=np.float32)
    a_tg = Tensor(a_np).realize()
    assert_close(det(a_tg), np.linalg.det(a_np))

@assert_one_kernel
def test_det_simple():
    a_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
    a_tg = Tensor(a_np).realize()
    assert_close(det(a_tg), np.linalg.det(a_np))

@assert_one_kernel
def test_det_random():
    np.random.seed(42)
    a_np = np.random.randn(4, 4).astype(np.float32)
    a_tg = Tensor(a_np).realize()
    assert_close(det(a_tg), np.linalg.det(a_np))

@assert_one_kernel
def test_det_singular():
    a_np = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    a_tg = Tensor(a_np).realize()
    assert_close(det(a_tg), np.linalg.det(a_np), atol=1e-6).realize()
