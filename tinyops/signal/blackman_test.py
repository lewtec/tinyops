from tinygrad import Tensor
import scipy.signal.windows as windows
from tinyops.signal.blackman import blackman
from tinyops._core import assert_one_kernel
from tinyops._core import assert_close

@assert_one_kernel
def test_blackman():
    M = 10
    result = blackman(M).realize()
    expected = windows.blackman(M)
    assert_close(result, expected)

    M = 11
    result = blackman(M).realize()
    expected = windows.blackman(M)
    assert_close(result, expected)

    M = 10
    result = blackman(M, sym=False).realize()
    expected = windows.blackman(M, sym=False)
    assert_close(result, expected)

    M = 11
    result = blackman(M, sym=False).realize()
    expected = windows.blackman(M, sym=False)
    assert_close(result, expected)
