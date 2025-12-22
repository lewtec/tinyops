import numpy as np; from tinygrad import Tensor; from tinyops.linalg.solve import solve; from tinyops._core import assert_close
@assert_one_kernel
def test_solve():
    np.random.seed(42); a_np = np.random.randn(5, 5).astype(np.float32) + np.eye(5)*5; b_np = np.random.randn(5).astype(np.float32)
    assert_close(solve(Tensor(a_np), Tensor(b_np)), np.linalg.solve(a_np, b_np), atol=1e-3).realize()
