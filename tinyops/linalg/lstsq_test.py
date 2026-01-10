import numpy as np
from tinygrad import Tensor
from tinyops.linalg.lstsq import lstsq
from tinyops._core import assert_close
from tinyops._core import assert_one_kernel

@assert_one_kernel
def test_lstsq():
    np.random.seed(42)
    a_np = np.random.randn(5, 3).astype(np.float32)
    b_np = np.random.randn(5).astype(np.float32)
    
    a = Tensor(a_np).realize()
    b = Tensor(b_np).realize()
    
    result = lstsq(a, b).realize()
    expected = np.linalg.lstsq(a_np, b_np, rcond=None)[0]
    
    assert_close(result, expected, atol=1e-3)
