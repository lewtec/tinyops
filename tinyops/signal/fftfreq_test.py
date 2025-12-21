import numpy as np
import unittest
from tinyops.signal.fftfreq import fftfreq
from tinyops._core import assert_close

class TestFFTFreq(unittest.TestCase):
  def test_fftfreq(self):
    n = 16
    d = 0.1
    y_np = np.fft.fftfreq(n, d)
    y_to = fftfreq(n, d).numpy()
    assert_close(y_to, y_np)

  def test_fftfreq_even(self):
    n = 10
    d = 1.0
    y_np = np.fft.fftfreq(n, d)
    y_to = fftfreq(n, d).numpy()
    assert_close(y_to, y_np)

  def test_fftfreq_odd(self):
    n = 11
    d = 1.0
    y_np = np.fft.fftfreq(n, d)
    y_to = fftfreq(n, d).numpy()
    assert_close(y_to, y_np)
