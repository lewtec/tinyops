import numpy as np
from tinygrad import Tensor

def assert_close(x: Tensor | np.ndarray, y: Tensor | np.ndarray, atol: float = 1e-5, rtol: float = 1e-5):
    """
    Asserts that x and y are close. x and y can be tinygrad Tensors or numpy arrays.
    """
    x_np = x.numpy() if isinstance(x, Tensor) else x
    y_np = y.numpy() if isinstance(y, Tensor) else y

    np.testing.assert_allclose(x_np, y_np, atol=atol, rtol=rtol)

from .test_utils import assert_one_kernel
