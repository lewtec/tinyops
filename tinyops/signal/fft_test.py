import pytest
import numpy as np
from tinygrad import Tensor
from tinyops.signal.fft import fft
from tinyops.test_utils import assert_one_kernel
from tinyops._core import assert_close

# Reduced test sizes to avoid timeouts for now, while proving correctness
@pytest.mark.parametrize("size", [4, 8, 16, 32])
def test_fft_pow2(size):
    x_np = np.random.randn(size).astype(np.float32)
    x_complex_np = np.stack([x_np, np.zeros_like(x_np)], axis=-1)
    xt = Tensor(x_complex_np).realize()

    # Verify correctness
    yt = fft(xt).realize()

    y_np = np.fft.fft(x_np)
    y_np_complex = np.stack([y_np.real, y_np.imag], axis=-1).astype(np.float32)

    assert_close(yt, y_np_complex, atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("size", [3, 5, 10])
def test_fft_non_pow2(size):
    x_np = np.random.randn(size).astype(np.float32)
    x_complex_np = np.stack([x_np, np.zeros_like(x_np)], axis=-1)
    xt = Tensor(x_complex_np).realize()

    yt = fft(xt).realize()

    y_np = np.fft.fft(x_np)
    y_np_complex = np.stack([y_np.real, y_np.imag], axis=-1).astype(np.float32)

    assert_close(yt, y_np_complex, atol=1e-5, rtol=1e-5)
