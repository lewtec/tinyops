import numpy as np
import pytest
from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.signal.fftfreq import fftfreq

def test_fftfreq():
    n = 10
    d = 0.5

    result_tg = fftfreq(n, d)
    result_np = np.fft.fftfreq(n, d)

    assert_close(result_tg, result_np)
