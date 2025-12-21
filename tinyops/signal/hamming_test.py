import numpy as np
import pytest
from tinyops.signal.hamming import hamming
from tinyops._core import assert_close

@pytest.mark.parametrize("M", [1, 10, 100])
def test_hamming(M):
    result = hamming(M)
    expected = np.hamming(M)
    assert_close(result, expected)
