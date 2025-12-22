import numpy as np
from tinygrad import Tensor
from tinyops.signal.ifft import ifft
from tinyops.test_utils import assert_one_kernel
from tinyops._core import assert_close

@assert_one_kernel
def test_ifft():
  x = np.random.randn(16) + 1j * np.random.randn(16)
  x_complex = np.stack([x.real, x.imag], axis=1)

  xt = Tensor(x_complex).realize()
  yt = ifft(xt).realize()

  y_np = np.fft.ifft(x)
  y_np_complex = np.stack([y_np.real, y_np.imag], axis=1)

  assert_close(yt, y_np_complex, atol=1e-6, rtol=1e-6)
