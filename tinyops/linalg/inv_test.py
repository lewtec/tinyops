import numpy as np
from tinygrad import Tensor

from tinyops._core import assert_close, assert_one_kernel
from tinyops.linalg.inv import inv


@assert_one_kernel
def test_inv():
    np.random.seed(42)
    a_np = np.random.randn(5, 5).astype(np.float32) + np.eye(5) * 5
    a = Tensor(a_np).realize()

    result = inv(a).realize()
    expected = np.linalg.inv(a_np)

    assert_close(result, expected, atol=1e-4)
