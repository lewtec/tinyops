from tinygrad import Tensor

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
    conjugate = Tensor.stack([complex_signal[:, 0], -complex_signal[:, 1]], dim=1)
    transformed = discrete_fourier_transform(conjugate)
    result_conjugate = Tensor.stack([transformed[:, 0], -transformed[:, 1]], dim=1)
    return result_conjugate / sample_count
