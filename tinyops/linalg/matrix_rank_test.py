import numpy as np
from tinygrad import Tensor
from tinyops.linalg.matrix_rank import matrix_rank
from tinyops.test_utils import assert_one_kernel

@assert_one_kernel
def test_matrix_rank():
    # Test with a full rank matrix
    a_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
    a_tiny = Tensor(a_np).realize()
    rank_tiny = matrix_rank(a_tiny)
    rank_np = np.linalg.matrix_rank(a_np)
    assert rank_tiny.item() == rank_np

    # Test with a rank deficient matrix
    b_np = np.array([[1, 2], [2, 4]], dtype=np.float32)
    b_tiny = Tensor(b_np).realize()
    rank_tiny = matrix_rank(b_tiny)
    rank_np = np.linalg.matrix_rank(b_np)
    assert rank_tiny.item() == rank_np

    # Test with a zero matrix
    c_np = np.zeros((3, 3), dtype=np.float32)
    c_tiny = Tensor(c_np).realize()
    rank_tiny = matrix_rank(c_tiny)
    rank_np = np.linalg.matrix_rank(c_np)
    assert rank_tiny.item() == rank_np

    # Test with a tall matrix
    d_np = np.random.randn(5, 3).astype(np.float32)
    d_tiny = Tensor(d_np).realize()
    rank_tiny = matrix_rank(d_tiny)
    rank_np = np.linalg.matrix_rank(d_np)
    assert rank_tiny.item() == rank_np

    # Test with a wide matrix
    e_np = np.random.randn(3, 5).astype(np.float32)
    e_tiny = Tensor(e_np).realize()
    rank_tiny = matrix_rank(e_tiny)
    rank_np = np.linalg.matrix_rank(e_np)
    assert rank_tiny.item() == rank_np
