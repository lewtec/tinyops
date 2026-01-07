import numpy as np
from tinygrad import Tensor, dtypes

def fft_cooley_tukey(x: Tensor) -> Tensor:
  """Computes FFT using Cooley-Tukey algorithm for power-of-two sizes."""
  N = x.shape[0]
  if N <= 1:
    return x

  # Split into even and odd
  even = fft_cooley_tukey(x[0::2])
  odd = fft_cooley_tukey(x[1::2])

  # Compute twiddle factors
  # We can compute these once and reuse if we wanted to optimize further,
  # but for now let's just make it correct.
  k = Tensor.arange(N // 2, dtype=dtypes.float32)
  angle = -2 * np.pi * k / N

  # complex exponential: cos(angle) + j*sin(angle)
  # x is (N, 2) where last dim is (real, imag)
  cos_a = angle.cos()
  sin_a = angle.sin()

  # odd * twiddle
  # (a + jb) * (c + jd) = (ac - bd) + j(ad + bc)
  # odd is (N/2, 2)
  # twiddle is (N/2, 2) implicitly (cos_a, sin_a)

  real = odd[:, 0] * cos_a - odd[:, 1] * sin_a
  imag = odd[:, 0] * sin_a + odd[:, 1] * cos_a
  odd_twiddle = Tensor.stack([real, imag], dim=1)

  # Combine
  return Tensor.cat(even + odd_twiddle, even - odd_twiddle, dim=0)

def fft(x: Tensor) -> Tensor:
  """
  Computes the one-dimensional discrete Fourier Transform.
  Args:
    x: The input tensor, representing complex numbers as a tensor with shape (N, 2).
  Returns:
    The FFT of the input tensor.
  """
  N = x.shape[0]
  if N <= 1:
    return x

  # Check if power of 2
  if (N & (N - 1)) == 0:
    return fft_cooley_tukey(x)
  else:
    # Fallback for non-power-of-2: Matrix Multiplication (DFT Matrix)
    # This is O(N^2) but avoids the complexity of Bluestein for now,
    # and might be faster for small non-pow-2 N in tinygrad than a deep recursion.
    # Construct DFT matrix
    k = Tensor.arange(N, dtype=dtypes.float32).unsqueeze(1)
    n = Tensor.arange(N, dtype=dtypes.float32).unsqueeze(0)
    angle = -2 * np.pi * k * n / N

    # W matrix (N, N, 2)
    W_real = angle.cos()
    W_imag = angle.sin()

    # x is (N, 2) -> (N, 1, 2) for broadcasting?
    # No, simple matmul if we handle complex correctly.
    # (a + jb) * (C + jD) = (aC - bD) + j(aD + bC)
    # We want W @ x
    # W_real @ x_real - W_imag @ x_imag
    # W_real @ x_imag + W_imag @ x_real

    x_real = x[:, 0]
    x_imag = x[:, 1]

    res_real = W_real @ x_real - W_imag @ x_imag
    res_imag = W_real @ x_imag + W_imag @ x_real

    return Tensor.stack([res_real, res_imag], dim=1)
