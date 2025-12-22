import numpy as np
from tinygrad import Tensor
from tinyops.signal.fft import fft
from tinyops.test_utils import assert_one_kernel
from tinyops._core import assert_close

@assert_one_kernel
def test_fft():
    x = np.random.randn(16)
    x_complex = np.stack([x, np.zeros_like(x)], axis=1)

    xt = Tensor(x_complex).realize()
    yt = fft(xt).realize()

    y_np = np.fft.fft(x)
    y_np_complex = np.stack([y_np.real, y_np.imag], axis=1)

    assert_close(yt, y_np_complex)
