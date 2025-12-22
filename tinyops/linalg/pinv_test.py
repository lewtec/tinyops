import numpy as np; from tinygrad import Tensor; from tinyops.linalg.pinv import pinv; from tinyops._core import assert_close
@assert_one_kernel
def test_pinv():
    np.random.seed(42); a_np = np.random.randn(5, 3).astype(np.float32)
    assert_close(pinv(Tensor(a_np)), np.linalg.pinv(a_np), atol=1e-4)
