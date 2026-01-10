import pytest
import scipy.signal.windows as windows
from tinyops.signal.hanning import hanning
from tinyops._core import assert_close
from tinyops._core import assert_one_kernel

@pytest.mark.parametrize("M, sym", [(10, True), (11, True), (10, False), (11, False)])
@assert_one_kernel
def test_hanning(M, sym):
    result = hanning(M, sym=sym).realize()
    expected = windows.hann(M, sym=sym)
    assert_close(result, expected)
