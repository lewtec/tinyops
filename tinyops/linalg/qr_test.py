import numpy as np
from tinygrad import Tensor

from tinyops._core import assert_close, assert_one_kernel
from tinyops.linalg.qr import qr


def is_upper_triangular(matrix: Tensor, atol=1e-5) -> bool:
    """Checks if a tinygrad Tensor is upper triangular."""
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return True
    mask = Tensor.ones(*matrix.shape, dtype=matrix.dtype).tril(-1)
    return (matrix * mask).abs().max().item() < atol


def run_qr_test(shape):
    """Helper function to run a QR test for a given matrix shape."""
    a_np = np.random.randn(*shape).astype(np.float32)
    # Ensure the matrix is not rank-deficient for stability
    if shape[0] == shape[1] and shape[0] > 0:
        a_np = a_np + np.eye(shape[0]) * 0.1
    a_tiny = Tensor(a_np).realize()

    q_tiny, r_tiny = qr(a_tiny)
    q_tiny.realize()
    r_tiny.realize()

    # 1. Check if Q is orthogonal (Q.T @ Q should be close to identity)
    k = min(shape)
    if k > 0:
        identity = Tensor.eye(k, dtype=a_tiny.dtype)
        assert_close(q_tiny.T @ q_tiny, identity, atol=1e-5, rtol=1e-5)

    # 2. Check if R is upper triangular
    assert is_upper_triangular(r_tiny), "R is not upper triangular"

    # 3. Check if Q @ R reconstructs A
    assert_close(q_tiny @ r_tiny, a_tiny, atol=1e-5, rtol=1e-5)


@assert_one_kernel
def test_qr():
    """Tests the QR decomposition for various matrix shapes."""
    run_qr_test((3, 3))  # Square matrix
    run_qr_test((5, 3))  # Tall matrix
    run_qr_test((3, 5))  # Wide matrix
    run_qr_test((0, 0))  # Empty matrix
    run_qr_test((5, 0))  # Empty matrix with rows
    run_qr_test((0, 5))  # Empty matrix with columns
