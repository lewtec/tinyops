import numpy as np
import pytest
from tinygrad import Tensor
from tinyops.signal.fft import fft
from tinyops._core import assert_close

def test_fft():
    x = np.random.randn(16).astype(np.float32)
    result = fft(Tensor(x))
    expected = np.fft.fft(x)
    assert_close(result, Tensor(np.stack([expected.real, expected.imag], axis=-1)))

def test_fft_non_power_of_two():
    x = Tensor.randn(15)
    with pytest.raises(ValueError, match="FFT size must be a power of 2"):
        fft(x)
