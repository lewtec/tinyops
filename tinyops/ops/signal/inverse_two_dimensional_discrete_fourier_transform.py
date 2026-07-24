"""Inverse two-dimensional discrete Fourier transform."""

from tinygrad import Tensor

from tinyops.ops.signal._fourier import separable_two_dimensional_transform
from tinyops.ops.signal.inverse_discrete_fourier_transform import (
    inverse_discrete_fourier_transform,
)


def inverse_two_dimensional_discrete_fourier_transform(complex_spectrum: Tensor) -> Tensor:
    """Compute the inverse two-dimensional discrete Fourier transform.

    Separable implementation: one-dimensional inverse DFT along width, then
    height. Matches ``numpy.fft.ifft2`` for complex arrays packed as
    real/imag pairs.

    Args:
        complex_spectrum: Spectrum tensor of shape ``(H, W, 2)`` where the
            last dimension contains ``(real, imaginary)`` parts.

    Returns:
        Inverse DFT result tensor of shape ``(H, W, 2)``.

    Raises:
        ValueError: If ``complex_spectrum`` is not shaped ``(H, W, 2)``.
    """
    return separable_two_dimensional_transform(
        complex_spectrum,
        inverse_discrete_fourier_transform,
    )
