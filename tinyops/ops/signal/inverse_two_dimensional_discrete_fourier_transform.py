"""Inverse two-dimensional discrete Fourier transform."""

from tinygrad import Tensor

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
    if complex_spectrum.ndim != 3 or complex_spectrum.shape[-1] != 2:
        raise ValueError(
            f"complex_spectrum must have shape (H, W, 2), got {complex_spectrum.shape}"
        )

    height, width, _ = complex_spectrum.shape
    if height == 0 or width == 0:
        return complex_spectrum

    after_width = Tensor.stack(
        [
            inverse_discrete_fourier_transform(complex_spectrum[row_index])
            for row_index in range(height)
        ],
        dim=0,
    )
    return Tensor.stack(
        [
            inverse_discrete_fourier_transform(after_width[:, column_index, :])
            for column_index in range(width)
        ],
        dim=1,
    )
