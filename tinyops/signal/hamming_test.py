from tinygrad import Tensor
import scipy.signal.windows as windows
from tinyops.signal.hamming import hamming
from tinyops.test_utils import assert_one_kernel
from tinyops._core import assert_close

@assert_one_kernel
def test_hamming():
    M = 10
    result = hamming(M).realize()
    expected = windows.hamming(M)
    assert_close(result, expected)

    M = 11
    result = hamming(M).realize()
    expected = windows.hamming(M)
    assert_close(result, expected)

    M = 10
    result = hamming(M, sym=False).realize()
    expected = windows.hamming(M, sym=False)
    assert_close(result, expected)

    M = 11
    result = hamming(M, sym=False).realize()
    expected = windows.hamming(M, sym=False)
    assert_close(result, expected)
