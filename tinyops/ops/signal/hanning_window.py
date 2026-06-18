import math

from tinygrad import Tensor


def hanning_window(length: int, symmetric: bool = True) -> Tensor:
    """Generate a Hanning (Hann) window.

    Args:
        length: Number of points in the window.
        symmetric: If True, generate a symmetric window for filter design.
            If False, generate a periodic window for spectral analysis.

    Returns:
        Tensor containing the window values.
    """
    if length < 1:
        return Tensor([])
    if length == 1:
        return Tensor.ones(1)

    denominator = length - 1 if symmetric else length
    indices = Tensor.arange(length)
    return 0.5 - 0.5 * (2 * math.pi * indices / denominator).cos()
