import numpy as np
import pytest
from tinyops.signal.hanning import hanning
from tinyops._core import assert_close

@pytest.mark.parametrize("M", [1, 10, 100])
def test_hanning(M):
    result = hanning(M)
    expected = np.hanning(M)
    assert_close(result, expected)
