import numpy as np
from tinygrad import Tensor

def fft(x: Tensor) -> Tensor:
  """
  Computes the one-dimensional discrete Fourier Transform.

  This function computes the FFT of a tensor whose size is a power of 2.

  Args:
    x: The input tensor.

  Returns:
    The FFT of the input tensor.
  """
  # Cooley-Tukey FFT algorithm
  N = x.shape[0]
  if N <= 1:
    return x

  # check that N is a power of 2
  if N & (N - 1) != 0:
    raise ValueError("FFT size must be a power of 2")

  even = fft(x[0::2])
  odd = fft(x[1::2])

  # Correctly calculate twiddle factors
  theta = Tensor.arange(N // 2) * -2 * np.pi / N
  T = Tensor.stack([theta.cos(), theta.sin()], dim=1)

  # Complex multiplication: odd * T
  odd_transformed = Tensor.stack([
    odd[:, 0] * T[:, 0] - odd[:, 1] * T[:, 1],
    odd[:, 0] * T[:, 1] + odd[:, 1] * T[:, 0]
  ], dim=1)

  # Combine results
  res_first_half = Tensor.stack([
      even[:, 0] + odd_transformed[:, 0],
      even[:, 1] + odd_transformed[:, 1]
  ], dim=1)

  res_second_half = Tensor.stack([
      even[:, 0] - odd_transformed[:, 0],
      even[:, 1] - odd_transformed[:, 1]
  ], dim=1)

  return Tensor.cat(res_first_half, res_second_half, dim=0)
