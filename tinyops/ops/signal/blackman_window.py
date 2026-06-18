import math

from tinygrad import Tensor


def blackman_window(length: int, symmetric: bool = True) -> Tensor:
    """Generate a Blackman window.

    Args:
        length: Number of points in the window.
        symmetric: If True, generate a symmetric window.

    Returns:
        Tensor containing the window values.
    """
    if length < 1:
        return Tensor([])
    if length == 1:
        return Tensor.ones(1)

    denominator = length - 1 if symmetric else length
    indices = Tensor.arange(length)
    return (
        0.42
        - 0.5 * (2 * math.pi * indices / denominator).cos()
        + 0.08 * (4 * math.pi * indices / denominator).cos()
    )
