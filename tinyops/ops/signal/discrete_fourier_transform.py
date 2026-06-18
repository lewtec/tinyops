import math

from tinygrad import Tensor, dtypes


def _cooley_tukey_fft(complex_signal: Tensor) -> Tensor:
    """Recursive Cooley-Tukey FFT for power-of-two lengths."""
    sample_count = complex_signal.shape[0]
    if sample_count <= 1:
        return complex_signal

    even = _cooley_tukey_fft(complex_signal[0::2])
    odd = _cooley_tukey_fft(complex_signal[1::2])

    frequency_indices = Tensor.arange(sample_count // 2, dtype=dtypes.float32)
    angle = -2 * math.pi * frequency_indices / sample_count

    cosine = angle.cos()
    sine = angle.sin()

    real_part = odd[:, 0] * cosine - odd[:, 1] * sine
    imaginary_part = odd[:, 0] * sine + odd[:, 1] * cosine
    twiddle_product = Tensor.stack([real_part, imaginary_part], dim=1)

    return Tensor.cat(even + twiddle_product, even - twiddle_product, dim=0)


def discrete_fourier_transform(complex_signal: Tensor) -> Tensor:
    """Compute the one-dimensional discrete Fourier transform.

    Automatically selects the Cooley-Tukey algorithm for power-of-two
    input sizes, or falls back to the DFT matrix for arbitrary sizes.

    Args:
        complex_signal: Input tensor of shape (N, 2) where the last
            dimension contains (real, imaginary) parts.

    Returns:
        DFT result tensor of shape (N, 2).
    """
    sample_count = complex_signal.shape[0]
    if sample_count <= 1:
        return complex_signal

    is_power_of_two = (sample_count & (sample_count - 1)) == 0
    if is_power_of_two:
        return _cooley_tukey_fft(complex_signal)

    # Fallback: dense DFT matrix multiplication (O(N^2))
    row_indices = Tensor.arange(sample_count, dtype=dtypes.float32).unsqueeze(1)
    column_indices = Tensor.arange(sample_count, dtype=dtypes.float32).unsqueeze(0)
    angle = -2 * math.pi * row_indices * column_indices / sample_count

    weight_real = angle.cos()
    weight_imaginary = angle.sin()

    input_real = complex_signal[:, 0]
    input_imaginary = complex_signal[:, 1]

    result_real = weight_real @ input_real - weight_imaginary @ input_imaginary
    result_imaginary = weight_real @ input_imaginary + weight_imaginary @ input_real

    return Tensor.stack([result_real, result_imaginary], dim=1)
