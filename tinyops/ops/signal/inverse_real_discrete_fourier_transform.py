"""Inverse of the one-dimensional real discrete Fourier transform."""

from tinygrad import Tensor, dtypes

from tinyops.ops.signal._fourier import complex_conjugate
from tinyops.ops.signal.inverse_discrete_fourier_transform import inverse_discrete_fourier_transform


def _resolve_reconstruction_length(spectrum_bin_count: int, length: int | None) -> int:
    """Resolve the real signal length ``N`` for an irfft-style reconstruction.

    When *length* is omitted, uses the ``numpy.fft.irfft`` default
    ``N = 2 * (M - 1)`` for ``M`` spectrum bins (even original length).
    Empty spectra map to length 0.
    """
    if length is not None:
        return length
    if spectrum_bin_count == 0:
        return 0
    return 2 * (spectrum_bin_count - 1)


def _resize_spectrum_bins(spectrum: Tensor, expected_bin_count: int) -> Tensor:
    """Pad or truncate packed ``(M, 2)`` spectrum bins to *expected_bin_count*.

    Matches ``numpy.fft.irfft``: missing high-frequency bins are zero-filled;
    extra bins are dropped.
    """
    spectrum_bin_count = spectrum.shape[0]
    if spectrum_bin_count < expected_bin_count:
        padding = Tensor.zeros(expected_bin_count - spectrum_bin_count, 2, dtype=dtypes.float32)
        return Tensor.cat(spectrum, padding, dim=0)
    if spectrum_bin_count > expected_bin_count:
        return spectrum[:expected_bin_count]
    return spectrum


def _complete_hermitian_spectrum(spectrum: Tensor, length: int) -> Tensor:
    """Build a length-``N`` complex spectrum by conjugating the positive bins.

    *spectrum* must already hold ``N//2+1`` non-negative frequency bins packed
    as ``(real, imag)``. Negative frequencies are filled so
    ``full[k] = conj(spectrum[N - k])``.
    """
    expected_bin_count = spectrum.shape[0]
    negative_frequency_count = length - expected_bin_count
    if negative_frequency_count == 0:
        return spectrum

    # full[k] = conj(spectrum[N - k]) for k = M .. N-1
    # → reverse of spectrum[1 : negative_frequency_count + 1], imag negated.
    mirrored = spectrum[1 : negative_frequency_count + 1].flip(0)
    conjugate_mirror = complex_conjugate(mirrored)
    return Tensor.cat(spectrum, conjugate_mirror, dim=0)


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

    length = _resolve_reconstruction_length(spectrum.shape[0], length)
    if length < 0:
        raise ValueError(f"length must be non-negative, got {length}")
    if length == 0:
        return Tensor.zeros(0, dtype=dtypes.float32)

    expected_bin_count = length // 2 + 1
    spectrum = _resize_spectrum_bins(spectrum.cast(dtypes.float32), expected_bin_count)
    full_spectrum = _complete_hermitian_spectrum(spectrum, length)
    recovered = inverse_discrete_fourier_transform(full_spectrum)
    return recovered[:, 0]
