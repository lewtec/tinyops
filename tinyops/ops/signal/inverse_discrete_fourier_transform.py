from tinygrad import Tensor

from tinyops.ops.signal._fourier import complex_conjugate
from tinyops.ops.signal.discrete_fourier_transform import discrete_fourier_transform


def inverse_discrete_fourier_transform(complex_signal: Tensor) -> Tensor:
    """Compute the one-dimensional inverse discrete Fourier transform.

    Args:
        complex_signal: Input tensor of shape (N, 2) where the last
            dimension contains (real, imaginary) parts.

    Returns:
        Inverse DFT result tensor of shape (N, 2).
    """
    sample_count = complex_signal.shape[0]
    transformed = discrete_fourier_transform(complex_conjugate(complex_signal))
    return complex_conjugate(transformed) / sample_count
