import numpy as np; from tinygrad import Tensor; from tinyops.linalg.lstsq import lstsq; from tinyops._core import assert_close
@assert_one_kernel
def test_lstsq():
    np.random.seed(42); a_np = np.random.randn(5, 3).astype(np.float32); b_np = np.random.randn(5).astype(np.float32)
    assert_close(lstsq(Tensor(a_np), Tensor(b_np)), np.linalg.lstsq(a_np, b_np, rcond=None)[0], atol=1e-3)
