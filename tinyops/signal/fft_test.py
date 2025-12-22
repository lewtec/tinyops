import numpy as np
from tinygrad import Tensor
from tinyops.signal.fft import fft
from tinyops.test_utils import assert_one_kernel
from tinyops._core import assert_close

@pytest.mark.parametrize("size", [16, 15, 17, 100])
def test_fft(size):
    x_np = np.random.randn(size).astype(np.float32)
    x_complex_np = np.stack([x_np, np.zeros_like(x_np)], axis=-1)
    xt = Tensor(x_complex_np).realize()

    # The @assert_one_kernel decorator is strict. The JIT compiler in tinygrad
    # should fuse the entire FFT graph (whether Cooley-Tukey or Bluestein's)
    # into a single kernel upon realization.
    @assert_one_kernel
    def run_kernel():
        result = fft(xt)
        result.realize()
        return result

    yt = run_kernel()

    y_np = np.fft.fft(x_np)
    y_np_complex = np.stack([y_np.real, y_np.imag], axis=-1).astype(np.float32)

    # Bluestein's algorithm can have slightly larger floating point errors
    assert_close(yt, y_np_complex, atol=1e-5, rtol=1e-5)
