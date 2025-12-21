import numpy as np
import unittest
from tinygrad import Tensor
from tinyops.signal.fft import fft, ifft
from tinyops._core import assert_close

class TestFFT(unittest.TestCase):
  def test_fft(self):
    x = np.random.randn(16)
    x_complex = np.stack([x, np.zeros_like(x)], axis=1)

    xt = Tensor(x_complex)
    yt = fft(xt).numpy()

    y_np = np.fft.fft(x)
    y_np_complex = np.stack([y_np.real, y_np.imag], axis=1)

    assert_close(yt, y_np_complex)

  def test_ifft(self):
    x = np.random.randn(16) + 1j * np.random.randn(16)
    x_complex = np.stack([x.real, x.imag], axis=1)

    xt = Tensor(x_complex)
    yt = ifft(xt).numpy()

    y_np = np.fft.ifft(x)
    y_np_complex = np.stack([y_np.real, y_np.imag], axis=1)

    assert_close(yt, y_np_complex, atol=1e-6, rtol=1e-6)
