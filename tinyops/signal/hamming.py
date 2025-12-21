from tinygrad import Tensor
import math

def hamming(M: int) -> Tensor:
    """
    Return the Hamming window.

    Parameters
    ----------
    M : int
        Number of points in the output window.

    Returns
    -------
    w : Tensor
        The window, with the maximum value normalized to 1.
    """
    if M == 1:
        return Tensor.ones(1)
    n = Tensor.arange(M)
    return 0.54 - 0.46 * (2 * math.pi * n / (M - 1)).cos()
