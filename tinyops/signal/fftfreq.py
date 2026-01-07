import numpy as np
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
    results = np.empty(n, dtype=int)
    N = (n - 1) // 2 + 1
    p1 = np.arange(0, N, dtype=int)
    results[:N] = p1
    p2 = np.arange(-(n // 2), 0, dtype=int)
    results[N:] = p2
    return Tensor(results * val, dtype=dtypes.float32)
