import numpy as np
from tinygrad import Tensor
from tinyops.signal.ifft import ifft
from tinyops._core import assert_close
import pytest

@pytest.mark.parametrize("size", [4, 8, 16])
def test_ifft(size):
  x = np.random.randn(size) + 1j * np.random.randn(size)
  x_complex = np.stack([x.real, x.imag], axis=1).astype(np.float32)

  xt = Tensor(x_complex).realize()
  yt = ifft(xt).realize()

  y_np = np.fft.ifft(x)
  y_np_complex = np.stack([y_np.real, y_np.imag], axis=1).astype(np.float32)

  assert_close(yt, y_np_complex, atol=1e-5, rtol=1e-5)
