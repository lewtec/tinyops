from tinygrad import Tensor, dtypes


def fourier_frequencies(window_length: int, sample_spacing: float = 1.0) -> Tensor:
    """Return the discrete Fourier transform sample frequencies.

    Args:
        window_length: Number of samples in the window.
        sample_spacing: Sample spacing (inverse of the sampling rate).

    Returns:
        Tensor of sample frequencies.
    """
    frequency_unit = 1.0 / (window_length * sample_spacing)
    positive_count = (window_length - 1) // 2 + 1
    negative_count = window_length // 2

    positive_frequencies = Tensor.arange(positive_count, dtype=dtypes.float32) * frequency_unit
    negative_frequencies = (Tensor.arange(negative_count, dtype=dtypes.float32) - negative_count) * frequency_unit

    return Tensor.cat(positive_frequencies, negative_frequencies)
