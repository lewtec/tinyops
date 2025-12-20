import numpy as np; from tinygrad import Tensor; from tinyops.linalg.inv import inv; from tinyops._core import assert_close
def test_inv():
    np.random.seed(42); a_np = np.random.randn(5, 5).astype(np.float32) + np.eye(5)*5
    assert_close(inv(Tensor(a_np)), np.linalg.inv(a_np), atol=1e-4)
