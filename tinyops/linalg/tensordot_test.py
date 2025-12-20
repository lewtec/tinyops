import numpy as np; from tinygrad import Tensor; from tinyops.linalg.tensordot import tensordot; from tinyops._core import assert_close
def test_tensordot_default():
    a = Tensor.randn(3, 4, 5); b = Tensor.randn(4, 5, 2)
    assert_close(tensordot(a, b), np.tensordot(a.numpy(), b.numpy()))
