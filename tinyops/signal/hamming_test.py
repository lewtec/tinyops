from tinygrad import Tensor
import scipy.signal.windows as windows
from tinyops.signal.hamming import hamming
from tinyops._core import assert_close

def test_hamming():
    M = 10
    result = hamming(M)
    expected = windows.hamming(M)
    assert_close(result, expected)

    M = 11
    result = hamming(M)
    expected = windows.hamming(M)
    assert_close(result, expected)

    M = 10
    result = hamming(M, sym=False)
    expected = windows.hamming(M, sym=False)
    assert_close(result, expected)

    M = 11
    result = hamming(M, sym=False)
    expected = windows.hamming(M, sym=False)
    assert_close(result, expected)
