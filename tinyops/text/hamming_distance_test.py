import pytest
import numpy as np
from tinygrad import Tensor
from tinyops.text.hamming_distance import hamming_distance
from tinyops._core import assert_close
from tinyops.test_utils import assert_one_kernel
from scipy.spatial.distance import hamming

@pytest.mark.parametrize("a_np, b_np", [
    (np.random.randint(0, 2, size=(10,)).astype(np.float32), np.random.randint(0, 2, size=(10,)).astype(np.float32)),
    (np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32), np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32)),
    (np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32), np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)),
])
def test_hamming_distance(a_np, b_np):
    a = Tensor(a_np).realize()
    b = Tensor(b_np).realize()

    @assert_one_kernel
    def run_kernel():
        result = hamming_distance(a, b)
        result.realize()
        return result

    result = run_kernel()
    expected = hamming(a_np, b_np)
    assert_close(result, expected)
