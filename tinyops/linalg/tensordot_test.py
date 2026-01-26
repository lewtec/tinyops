import numpy as np
from tinygrad import Tensor

from tinyops._core import assert_close, assert_one_kernel
from tinyops.linalg.tensordot import tensordot


@assert_one_kernel
def test_tensordot_default():
    a = Tensor.randn(3, 4, 5).realize()
    b = Tensor.randn(4, 5, 2).realize()

    result = tensordot(a, b).realize()
    expected = np.tensordot(a.numpy(), b.numpy())

    assert_close(result, expected)
