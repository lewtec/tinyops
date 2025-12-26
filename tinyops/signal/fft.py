import numpy as np
from tinygrad import Tensor

def fft_cooley_tukey(x: Tensor) -> Tensor:
  """Computes FFT using Cooley-Tukey algorithm for power-of-two sizes."""
  N = x.shape[0]
  if N <= 1:
    return x

  even = _fft_cooley_tukey(x[0::2])
  odd = _fft_cooley_tukey(x[1::2])

  theta = Tensor.arange(N // 2) * -2 * np.pi / N
  T = Tensor.stack([theta.cos(), theta.sin()], dim=1)

  odd_transformed = Tensor.stack([
    odd[:, 0] * T[:, 0] - odd[:, 1] * T[:, 1],
    odd[:, 0] * T[:, 1] + odd[:, 1] * T[:, 0]
  ], dim=1)

  res_first_half = Tensor.stack([
      even[:, 0] + odd_transformed[:, 0],
      even[:, 1] + odd_transformed[:, 1]
  ], dim=1)

  res_second_half = Tensor.stack([
      even[:, 0] - odd_transformed[:, 0],
      even[:, 1] - odd_transformed[:, 1]
  ], dim=1)

  return Tensor.cat(res_first_half, res_second_half, dim=0)


def fft_bluestein(x: Tensor) -> Tensor:
  """Computes FFT using Bluestein's algorithm for non-power-of-two sizes."""
  N = x.shape[0]
  M = 1 << (2 * N - 1).bit_length()

  # Chirp sequence
  k = Tensor.arange(N)
  theta = np.pi * k.pow(2) / N
  b = Tensor.stack([theta.cos(), theta.sin()], dim=1)
  b_conj = Tensor.stack([b[:, 0], -b[:, 1]], dim=1)

  # Input sequence, element-wise multiplied by b_conj
  a = Tensor.stack([
      x[:, 0] * b_conj[:, 0] - x[:, 1] * b_conj[:, 1],
      x[:, 0] * b_conj[:, 1] + x[:, 1] * b_conj[:, 0]
  ], dim=1)

  # Padded input sequence
  a_padded = a.pad(((0, M - N), (0, 0)))

  # Convolution kernel
  k_h_pos = Tensor.arange(N)
  theta_h_pos = np.pi * k_h_pos.pow(2) / N
  h_pos = Tensor.stack([theta_h_pos.cos(), theta_h_pos.sin()], dim=1)

  k_h_neg = Tensor.arange(1, N).flip(0)
  theta_h_neg = np.pi * k_h_neg.pow(2) / N
  h_neg = Tensor.stack([theta_h_neg.cos(), theta_h_neg.sin()], dim=1)

  h = Tensor.cat(h_pos, Tensor.zeros(M - (2 * N - 1), 2), h_neg) if N > 1 else h_pos

  # Convolution via FFT (recursively calls the main fft dispatcher)
  fft_a = fft(a_padded)
  fft_h = fft(h)

  conv_fft = Tensor.stack([
      fft_a[:, 0] * fft_h[:, 0] - fft_a[:, 1] * fft_h[:, 1],
      fft_a[:, 0] * fft_h[:, 1] + fft_a[:, 1] * fft_h[:, 0]
  ], dim=1)

  # Inverse FFT of the convolution
  conv_ifft = ifft(conv_fft)

  # Final result, element-wise multiplied by b_conj
  conv_result = conv_ifft[0:N]
  final_result = Tensor.stack([
      conv_result[:, 0] * b_conj[:, 0] - conv_result[:, 1] * b_conj[:, 1],
      conv_result[:, 0] * b_conj[:, 1] + conv_result[:, 1] * b_conj[:, 0]
  ], dim=1)

  return final_result


def fft(x: Tensor) -> Tensor:
  """
  Computes the one-dimensional discrete Fourier Transform.

  This function computes the FFT of a tensor. It uses the Cooley-Tukey algorithm
  for sizes that are a power of 2, and Bluestein's algorithm for other sizes.

  Args:
    x: The input tensor, representing complex numbers as a tensor with shape (N, 2).

  Returns:
    The FFT of the input tensor.
  """
  N = x.shape[0]
  if N <= 1:
    return x

  # Dispatch to the appropriate algorithm
  if (N & (N - 1)) == 0 and N != 0:
    return fft_cooley_tukey(x)
  else:
    return fft_bluestein(x)

