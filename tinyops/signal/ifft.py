from tinygrad import Tensor
from tinyops.signal.fft import fft

def ifft(x: Tensor) -> Tensor:
  """
  Computes the one-dimensional inverse discrete Fourier Transform.

  Args:
    x: The input tensor.

  Returns:
    The inverse FFT of the input tensor.
  """
  N = x.shape[0]

  # Conjugate the input
  x_conj = Tensor.stack([x[:, 0], -x[:, 1]], dim=1)

  # Apply FFT
  y = fft(x_conj)

  # Conjugate the result and scale
  y_conj = Tensor.stack([y[:, 0], -y[:, 1]], dim=1)

  return y_conj / N
