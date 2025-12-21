import numpy as np
from tinygrad import Tensor
from tinyops.signal.ifft import ifft
from tinyops._core import assert_close

def test_ifft():
    x = np.random.randn(16, 2).astype(np.float32)
    result = ifft(Tensor(x))
    expected = np.fft.ifft(x[..., 0] + 1j * x[..., 1])
    assert_close(result, Tensor(np.stack([expected.real, expected.imag], axis=-1)), atol=1e-5, rtol=1e-5)
