import numpy as np
from tinygrad import Tensor, dtypes
from tinyops.signal import fft
from tinyops._core import assert_close

def test_fft_simple():
    # Simple real signal
    x_real = Tensor([1, 2, 3, 4], dtype=dtypes.float32)

    # tinyops fft
    y = fft(x_real)

    # numpy fft
    y_np = np.fft.fft(x_real.numpy())

    # Create a complex tensor for comparison
    y_np_complex = Tensor(np.stack([y_np.real, y_np.imag], axis=-1), dtype=dtypes.float32)

    assert_close(y, y_np_complex)

def test_fft_complex():
    # Complex signal
    x_real = np.array([1, 2, 3, 4], dtype=np.float32)
    x_imag = np.array([5, 6, 7, 8], dtype=np.float32)
    x_complex = Tensor(np.stack([x_real, x_imag], axis=-1), dtype=dtypes.float32)

    # tinyops fft
    y = fft(x_complex)

    # numpy fft
    y_np = np.fft.fft(x_real + 1j * x_imag)

    # Create a complex tensor for comparison
    y_np_complex = Tensor(np.stack([y_np.real, y_np.imag], axis=-1), dtype=dtypes.float32)

    assert_close(y, y_np_complex)

def test_fft_non_power_of_two():
    # Test padding
    x_real = Tensor([1, 2, 3], dtype=dtypes.float32)

    # tinyops fft
    y = fft(x_real)

    # numpy fft (numpy pads internally for speed, but the result is for the original length.
    # My implementation pads to the next power of two, so I should compare with a padded numpy result)
    padded_x_np = np.pad(x_real.numpy(), (0, 1), 'constant')
    y_np = np.fft.fft(padded_x_np)

    y_np_complex = Tensor(np.stack([y_np.real, y_np.imag], axis=-1), dtype=dtypes.float32)

    assert_close(y, y_np_complex)
