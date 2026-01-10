import numpy as np
from tinygrad import Tensor
from tinyops.linalg.einsum import einsum
from tinyops._core import assert_close
from tinyops._core import assert_one_kernel

@assert_one_kernel
def test_einsum_matmul():
    a = Tensor.randn(3, 4).realize()
    b = Tensor.randn(4, 5).realize()
    
    result = einsum('ij,jk->ik', a, b).realize()
    expected = np.einsum('ij,jk->ik', a.numpy(), b.numpy())
    
    assert_close(result, expected)
