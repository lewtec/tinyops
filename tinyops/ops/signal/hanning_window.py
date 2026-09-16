from tinygrad import Tensor

from tinyops.ops.signal._window import HANNING_COEFFICIENTS, cosine_sum_window


def hanning_window(length: int, symmetric: bool = True) -> Tensor:
    """Generate a Hanning (Hann) window.

    Args:
        length: Number of points in the window.
        symmetric: If True, generate a symmetric window for filter design.
            If False, generate a periodic window for spectral analysis.

    Returns:
        Tensor containing the window values.
    """
    return cosine_sum_window(length, HANNING_COEFFICIENTS, symmetric=symmetric)
