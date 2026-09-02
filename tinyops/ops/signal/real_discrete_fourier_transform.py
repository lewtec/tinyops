"""One-dimensional real discrete Fourier transform (non-negative frequencies)."""

from tinygrad import Tensor, dtypes

from tinyops.ops.signal.discrete_fourier_transform import discrete_fourier_transform


def real_discrete_fourier_transform(real_signal: Tensor) -> Tensor:
    """Compute the 1D DFT of a real-valued signal.

    Only the non-negative frequency bins are returned. For a length-``N``
    input this is ``N // 2 + 1`` complex bins, matching ``numpy.fft.rfft``.

    Args:
        real_signal: Real input tensor of shape ``(N,)``.

    Returns:
        Spectrum tensor of shape ``(N // 2 + 1, 2)`` where the last axis is
        ``(real, imaginary)``.

    Raises:
        ValueError: If ``real_signal`` is not one-dimensional.
    """
    if real_signal.ndim != 1:
        raise ValueError(f"real_signal must be 1-D, got shape {real_signal.shape}")

    sample_count = real_signal.shape[0]
    if sample_count == 0:
        return Tensor.zeros(0, 2, dtype=dtypes.float32)

    real_part = real_signal.cast(dtypes.float32)
    imaginary_part = Tensor.zeros(sample_count, dtype=dtypes.float32)
    complex_signal = Tensor.stack([real_part, imaginary_part], dim=1)
    full_spectrum = discrete_fourier_transform(complex_signal)
    non_negative_bin_count = sample_count // 2 + 1
    return full_spectrum[:non_negative_bin_count]
