from tinygrad import Tensor

def fftfreq(n: int, d: float = 1.0) -> Tensor:
  """
  Computes the discrete Fourier Transform sample frequencies.

  Args:
    n: The number of samples.
    d: The sample spacing.

  Returns:
    A tensor containing the sample frequencies.
  """
  val = 1.0 / (n * d)
  p1 = Tensor.arange(0, (n - 1) // 2 + 1)
  p2 = Tensor.arange(-(n // 2), 0)
  results = Tensor.cat(p1, p2)
  return results * val
