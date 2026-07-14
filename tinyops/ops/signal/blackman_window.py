from tinygrad import Tensor

from tinyops.ops.signal._window import BLACKMAN_COEFFICIENTS, cosine_sum_window


def blackman_window(length: int, symmetric: bool = True) -> Tensor:
    """Generate a Blackman window.

    Args:
        length: Number of points in the window.
        symmetric: If True, generate a symmetric window.

    Returns:
        Tensor containing the window values.
    """
    return cosine_sum_window(length, BLACKMAN_COEFFICIENTS, symmetric=symmetric)
