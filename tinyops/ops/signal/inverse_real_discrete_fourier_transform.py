"""Inverse of the one-dimensional real discrete Fourier transform."""

from tinygrad import Tensor, dtypes

from tinyops.ops.signal.inverse_discrete_fourier_transform import inverse_discrete_fourier_transform


def inverse_real_discrete_fourier_transform(
    spectrum: Tensor,
    length: int | None = None,
) -> Tensor:
    """Reconstruct a real signal from its non-negative DFT bins.

    Completes the Hermitian spectrum and applies the inverse DFT. When
    ``length`` is omitted, an even original length ``N = 2 * (M - 1)`` is
    assumed (``numpy.fft.irfft`` default), where ``M`` is the number of
    spectrum bins.

    Args:
        spectrum: Complex spectrum of shape ``(M, 2)`` with ``(real,
            imaginary)`` columns, as returned by
            :func:`~tinyops.ops.signal.real_discrete_fourier_transform.real_discrete_fourier_transform`.
        length: Original real signal length ``N``. When ``None``, uses
            ``2 * (M - 1)``.

    Returns:
        Real tensor of shape ``(N,)``.

    Raises:
        ValueError: If ``spectrum`` is not shaped ``(M, 2)`` or ``length`` is
            negative.
    """
    if spectrum.ndim != 2 or spectrum.shape[1] != 2:
        raise ValueError(f"spectrum must have shape (M, 2), got {spectrum.shape}")

    spectrum_bin_count = spectrum.shape[0]
    if length is None:
        if spectrum_bin_count == 0:
            return Tensor.zeros(0, dtype=dtypes.float32)
        length = 2 * (spectrum_bin_count - 1)

    if length < 0:
        raise ValueError(f"length must be non-negative, got {length}")
    if length == 0:
        return Tensor.zeros(0, dtype=dtypes.float32)

    expected_bin_count = length // 2 + 1
    spectrum = spectrum.cast(dtypes.float32)

    # numpy.fft.irfft pads or truncates the spectrum to N//2+1 bins.
    if spectrum_bin_count < expected_bin_count:
        padding = Tensor.zeros(expected_bin_count - spectrum_bin_count, 2, dtype=dtypes.float32)
        spectrum = Tensor.cat(spectrum, padding, dim=0)
    elif spectrum_bin_count > expected_bin_count:
        spectrum = spectrum[:expected_bin_count]

    negative_frequency_count = length - expected_bin_count
    if negative_frequency_count == 0:
        full_spectrum = spectrum
    else:
        # full[k] = conj(spectrum[N - k]) for k = M .. N-1
        # → reverse of spectrum[1 : negative_frequency_count + 1], imag negated.
        mirrored = spectrum[1 : negative_frequency_count + 1].flip(0)
        conjugate_mirror = Tensor.stack([mirrored[:, 0], -mirrored[:, 1]], dim=1)
        full_spectrum = Tensor.cat(spectrum, conjugate_mirror, dim=0)

    recovered = inverse_discrete_fourier_transform(full_spectrum)
    return recovered[:, 0]
