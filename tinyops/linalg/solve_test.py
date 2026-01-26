import numpy as np
from tinygrad import Tensor

from tinyops._core import assert_close, assert_one_kernel
from tinyops.linalg.solve import solve


@assert_one_kernel
def test_solve():
    np.random.seed(42)
    a_np = np.random.randn(5, 5).astype(np.float32) + np.eye(5) * 5
    b_np = np.random.randn(5).astype(np.float32)

    a = Tensor(a_np).realize()
    b = Tensor(b_np).realize()

    result = solve(a, b).realize()
    expected = np.linalg.solve(a_np, b_np)

    assert_close(result, expected, atol=1e-3)
