"""Shared helpers for cosine-sum window generation."""

import math
from collections.abc import Sequence

from tinygrad import Tensor

# Standard cosine-sum coefficients (a0, a1, a2, ...).
# Window formula: sum_k (-1)^k * a_k * cos(2 * pi * k * n / denominator).
HANNING_COEFFICIENTS: tuple[float, ...] = (0.5, 0.5)
HAMMING_COEFFICIENTS: tuple[float, ...] = (0.54, 0.46)
BLACKMAN_COEFFICIENTS: tuple[float, ...] = (0.42, 0.5, 0.08)


def cosine_sum_window(
    length: int,
    coefficients: Sequence[float],
    *,
    symmetric: bool = True,
) -> Tensor:
    """Generate a cosine-sum window of the given length.

    Args:
        length: Number of points in the window.
        coefficients: Cosine-sum coefficients ``(a0, a1, ...)``.
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
    fundamental_angle = 2 * math.pi * indices / denominator

    result = Tensor.full((length,), float(coefficients[0]))
    for harmonic, coefficient in enumerate(coefficients[1:], start=1):
        term = float(coefficient) * (fundamental_angle * harmonic).cos()
        if harmonic % 2 == 1:
            result = result - term
        else:
            result = result + term
    return result
