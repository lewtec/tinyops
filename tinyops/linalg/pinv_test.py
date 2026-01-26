import numpy as np
from tinygrad import Tensor

from tinyops._core import assert_close, assert_one_kernel
from tinyops.linalg.pinv import pinv


@assert_one_kernel
def test_pinv():
    np.random.seed(42)
    a_np = np.random.randn(5, 3).astype(np.float32)
    a = Tensor(a_np).realize()

    result = pinv(a).realize()
    expected = np.linalg.pinv(a_np)

    assert_close(result, expected, atol=1e-4)
