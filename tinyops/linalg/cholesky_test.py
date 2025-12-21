import numpy as np
from tinygrad import Tensor
from tinyops.linalg.cholesky import cholesky
from tinyops._core import assert_close

def test_cholesky():
    # Create a symmetric positive-definite matrix
    a = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=np.float32)
    a_tiny = Tensor(a)

    # Compute Cholesky decomposition
    l_tiny = cholesky(a_tiny)
    l_np = np.linalg.cholesky(a)

    assert_close(l_tiny, l_np)

    # Test with another matrix
    a = np.random.rand(5, 5).astype(np.float32)
    a = np.dot(a, a.T) + np.eye(5) * 1e-3 # Ensure it's positive-definite
    a_tiny = Tensor(a)

    l_tiny = cholesky(a_tiny)
    l_np = np.linalg.cholesky(a)

    assert_close(l_tiny, l_np)
