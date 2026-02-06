from tinygrad import Tensor, dtypes


def fftfreq(n: int, d: float = 1.0) -> Tensor:
    """
    Return the Discrete Fourier Transform sample frequencies.

    Args:
        n: Window length.
        d: Sample spacing (inverse of the sampling rate).

    Returns:
        Tensor containing the sample frequencies.
    """
    val = 1.0 / (n * d)
    N = (n - 1) // 2 + 1
    p1 = Tensor.arange(N, dtype=dtypes.float32)
    p2 = Tensor.arange(-(n // 2), 0, dtype=dtypes.float32)
    results = Tensor.cat(p1, p2, dim=0)
    return results * val
