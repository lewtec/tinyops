"""Two-dimensional discrete Fourier transform."""

from tinygrad import Tensor

from tinyops.ops.signal.discrete_fourier_transform import discrete_fourier_transform


def two_dimensional_discrete_fourier_transform(complex_image: Tensor) -> Tensor:
    """Compute the two-dimensional discrete Fourier transform.

    Separable implementation: one-dimensional DFT along width, then height.
    Matches ``numpy.fft.fft2`` for complex arrays packed as real/imag pairs.

    Args:
        complex_image: Input tensor of shape ``(H, W, 2)`` where the last
            dimension contains ``(real, imaginary)`` parts.

    Returns:
        DFT result tensor of shape ``(H, W, 2)``.

    Raises:
        ValueError: If ``complex_image`` is not shaped ``(H, W, 2)``.
    """
    if complex_image.ndim != 3 or complex_image.shape[-1] != 2:
        raise ValueError(
            f"complex_image must have shape (H, W, 2), got {complex_image.shape}"
        )

    height, width, _ = complex_image.shape
    if height == 0 or width == 0:
        return complex_image

    after_width = Tensor.stack(
        [discrete_fourier_transform(complex_image[row_index]) for row_index in range(height)],
        dim=0,
    )
    return Tensor.stack(
        [
            discrete_fourier_transform(after_width[:, column_index, :])
            for column_index in range(width)
        ],
        dim=1,
    )
