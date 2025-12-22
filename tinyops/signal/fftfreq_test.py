import numpy as np
import pytest
from tinyops.signal.fftfreq import fftfreq
from tinyops.test_utils import assert_one_kernel
from tinyops._core import assert_close

TEST_PARAMS = [
    (16, 0.1),
    (10, 1.0),
    (11, 1.0),
]

@pytest.mark.parametrize("n,d", TEST_PARAMS)
@assert_one_kernel
def test_fftfreq(n, d):
    y_np = np.fft.fftfreq(n, d)
    y_to = fftfreq(n, d).realize()
    assert_close(y_to, y_np)
