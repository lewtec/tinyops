import numpy as np; from tinygrad import Tensor; from tinyops.linalg.einsum import einsum; from tinyops._core import assert_close
def test_einsum_matmul():
    a = Tensor.randn(3, 4); b = Tensor.randn(4, 5)
    assert_close(einsum('ij,jk->ik', a, b), np.einsum('ij,jk->ik', a.numpy(), b.numpy()))
def test_einsum_diagonal():
    a = Tensor.randn(3, 3)
    assert_close(einsum('ii->i', a), np.einsum('ii->i', a.numpy()))
