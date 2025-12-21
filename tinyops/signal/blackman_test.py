import numpy as np
import pytest
from tinyops.signal.blackman import blackman
from tinyops._core import assert_close

@pytest.mark.parametrize("M", [1, 10, 100])
def test_blackman(M):
    result = blackman(M)
    expected = np.blackman(M)
    assert_close(result, expected)
