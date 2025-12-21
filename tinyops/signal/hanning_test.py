from tinygrad import Tensor
import scipy.signal.windows as windows
from tinyops.signal.hanning import hanning
from tinyops._core import assert_close

def test_hanning():
    M = 10
    result = hanning(M)
    expected = windows.hann(M)
    assert_close(result, expected)

    M = 11
    result = hanning(M)
    expected = windows.hann(M)
    assert_close(result, expected)

    M = 10
    result = hanning(M, sym=False)
    expected = windows.hann(M, sym=False)
    assert_close(result, expected)

    M = 11
    result = hanning(M, sym=False)
    expected = windows.hann(M, sym=False)
    assert_close(result, expected)
