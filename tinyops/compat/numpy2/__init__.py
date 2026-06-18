"""numpy 2.x compatibility layer.

Provides numpy-compatible function signatures that delegate to tinyops.ops.
"""

from tinygrad import Tensor

from tinyops.ops.linear_algebra.cholesky_decomposition import cholesky_decomposition as _cholesky
from tinyops.ops.linear_algebra.condition_number import condition_number as _condition_number
from tinyops.ops.linear_algebra.determinant import determinant as _determinant
from tinyops.ops.linear_algebra.diagonal import diagonal as _diagonal
from tinyops.ops.linear_algebra.dot_product import dot_product as _dot_product
from tinyops.ops.linear_algebra.einstein_summation import einstein_summation as _einstein_summation
from tinyops.ops.linear_algebra.inner_product import inner_product as _inner_product
from tinyops.ops.linear_algebra.inverse import inverse as _inverse
from tinyops.ops.linear_algebra.kronecker_product import kronecker_product as _kronecker_product
from tinyops.ops.linear_algebra.least_squares import least_squares as _least_squares
from tinyops.ops.linear_algebra.matrix_multiply import matrix_multiply as _matrix_multiply
from tinyops.ops.linear_algebra.matrix_power import matrix_power as _matrix_power
from tinyops.ops.linear_algebra.matrix_rank import matrix_rank as _matrix_rank
from tinyops.ops.linear_algebra.norm import norm as _norm
from tinyops.ops.linear_algebra.outer_product import outer_product as _outer_product
from tinyops.ops.linear_algebra.pseudo_inverse import pseudo_inverse as _pseudo_inverse
from tinyops.ops.linear_algebra.qr_decomposition import qr_decomposition as _qr_decomposition
from tinyops.ops.linear_algebra.solve_linear_system import solve_linear_system as _solve
from tinyops.ops.linear_algebra.tensor_dot_product import tensor_dot_product as _tensor_dot_product
from tinyops.ops.linear_algebra.trace import trace as _trace
from tinyops.ops.linear_algebra.vector_dot_product import vector_dot_product as _vector_dot_product
from tinyops.ops.signal.blackman_window import blackman_window as _blackman_window
from tinyops.ops.signal.convolution_1d import ConvolutionMode
from tinyops.ops.signal.convolution_1d import convolution_1d as _convolution_1d
from tinyops.ops.signal.discrete_fourier_transform import discrete_fourier_transform as _dft
from tinyops.ops.signal.fourier_frequencies import fourier_frequencies as _fourier_frequencies
from tinyops.ops.signal.hamming_window import hamming_window as _hamming_window
from tinyops.ops.signal.hanning_window import hanning_window as _hanning_window
from tinyops.ops.signal.inverse_discrete_fourier_transform import inverse_discrete_fourier_transform as _idft
from tinyops.ops.statistics.arithmetic_mean import arithmetic_mean as _arithmetic_mean
from tinyops.ops.statistics.bin_count import bin_count as _bin_count
from tinyops.ops.statistics.correlation_coefficients import correlation_coefficients as _correlation_coefficients
from tinyops.ops.statistics.covariance_matrix import covariance_matrix as _covariance_matrix
from tinyops.ops.statistics.cross_correlation import CorrelationMode
from tinyops.ops.statistics.cross_correlation import cross_correlation as _cross_correlation
from tinyops.ops.statistics.digitize import digitize as _digitize
from tinyops.ops.statistics.histogram import histogram as _histogram
from tinyops.ops.statistics.histogram_2d import histogram_2d as _histogram_2d
from tinyops.ops.statistics.median import median as _median
from tinyops.ops.statistics.peak_to_peak import peak_to_peak as _peak_to_peak
from tinyops.ops.statistics.percentile import percentile as _percentile
from tinyops.ops.statistics.quantile import quantile as _quantile
from tinyops.ops.statistics.standard_deviation import standard_deviation as _standard_deviation
from tinyops.ops.statistics.variance import variance as _variance
from tinyops.ops.statistics.weighted_average import weighted_average as _weighted_average

# --- Statistics ---


def mean(a: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    """Compute the arithmetic mean along the specified axis."""
    return _arithmetic_mean(a, axis=axis, keep_dimensions=keepdims)


def median(a: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    """Compute the median along the specified axis."""
    return _median(a, axis=axis, keep_dimensions=keepdims)


def std(a: Tensor, axis=None, ddof: int = 0, keepdims: bool = False) -> Tensor:
    """Compute the standard deviation along the specified axis."""
    return _standard_deviation(a, axis=axis, degrees_of_freedom=ddof, keep_dimensions=keepdims)


def var(a: Tensor, axis=None, ddof: int = 0, keepdims: bool = False) -> Tensor:
    """Compute the variance along the specified axis."""
    return _variance(a, axis=axis, degrees_of_freedom=ddof, keep_dimensions=keepdims)


def average(a: Tensor, axis=None, weights: Tensor | None = None, returned: bool = False):
    """Compute the weighted average along the specified axis."""
    return _weighted_average(a, axis=axis, weights=weights, return_sum_of_weights=returned)


def percentile(a: Tensor, q, axis=None, keepdims: bool = False, method: str = "linear") -> Tensor:
    """Compute the q-th percentile of the data along the specified axis."""
    if method != "linear":
        raise NotImplementedError("Only 'linear' interpolation is supported")
    return _percentile(a, q, axis=axis, keep_dimensions=keepdims)


def quantile(a: Tensor, q, axis=None, keepdims: bool = False, method: str = "linear") -> Tensor:
    """Compute the q-th quantile of the data along the specified axis."""
    if method != "linear":
        raise NotImplementedError("Only 'linear' interpolation is supported")
    return _quantile(a, q, axis=axis, keep_dimensions=keepdims)


def ptp(a: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    """Range of values (maximum - minimum) along an axis."""
    return _peak_to_peak(a, axis=axis, keep_dimensions=keepdims)


def bincount(x: Tensor, weights: Tensor | None = None, minlength: int = 0) -> Tensor:
    """Count number of occurrences of each value in array of non-negative ints."""
    return _bin_count(x, weights=weights, minimum_output_length=minlength)


def digitize(x: Tensor, bins: Tensor, right: bool = False) -> Tensor:
    """Return the indices of the bins to which each value in input array belongs."""
    return _digitize(x, bins, right_closed=right)


def corrcoef(x: Tensor, y: Tensor | None = None, rowvar: bool = True) -> Tensor:
    """Return Pearson product-moment correlation coefficients."""
    return _correlation_coefficients(x, second_observations=y, rows_are_variables=rowvar)


def correlate(a: Tensor, v: Tensor, mode: str = "valid") -> Tensor:
    """Cross-correlation of two 1-dimensional sequences."""
    mode_map = {"valid": CorrelationMode.VALID, "same": CorrelationMode.SAME, "full": CorrelationMode.FULL}
    return _cross_correlation(a, v, mode=mode_map[mode])


def cov(m: Tensor, y: Tensor | None = None, rowvar: bool = True, ddof: int = 1) -> Tensor:
    """Estimate a covariance matrix."""
    return _covariance_matrix(m, second_observations=y, rows_are_variables=rowvar, degrees_of_freedom=ddof)


def histogram(a: Tensor, bins: int = 10, range: tuple[float, float] | None = None, density: bool = False):
    """Compute the histogram of a dataset."""
    return _histogram(a, number_of_bins=bins, value_range=range, compute_density=density)


def histogram2d(x: Tensor, y: Tensor, bins=10, range=None, density: bool = False):
    """Compute the bi-dimensional histogram of two data samples."""
    return _histogram_2d(x, y, number_of_bins=bins, value_range=range, compute_density=density)


# --- Linear Algebra (top-level) ---


def dot(a: Tensor, b: Tensor) -> Tensor:
    """Dot product of two arrays."""
    return _dot_product(a, b)


def matmul(x1: Tensor, x2: Tensor) -> Tensor:
    """Matrix product of two arrays."""
    return _matrix_multiply(x1, x2)


def vdot(a: Tensor, b: Tensor) -> Tensor:
    """Return the dot product of two vectors."""
    return _vector_dot_product(a, b)


def inner(a: Tensor, b: Tensor) -> Tensor:
    """Inner product of two arrays."""
    return _inner_product(a, b)


def outer(a: Tensor, b: Tensor) -> Tensor:
    """Compute the outer product of two vectors."""
    return _outer_product(a, b)


def tensordot(a: Tensor, b: Tensor, axes=2) -> Tensor:
    """Compute tensor dot product along specified axes."""
    return _tensor_dot_product(a, b, axes=axes)


def einsum(subscripts: str, *operands: Tensor) -> Tensor:
    """Evaluates the Einstein summation convention on the operands."""
    return _einstein_summation(subscripts, *operands)


def trace(a: Tensor, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Tensor:
    """Return the sum along diagonals of the array."""
    return _trace(a, offset=offset, axis_1=axis1, axis_2=axis2)


def diagonal(a: Tensor, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Tensor:
    """Return specified diagonals."""
    return _diagonal(a, offset=offset, axis_1=axis1, axis_2=axis2)


def kron(a: Tensor, b: Tensor) -> Tensor:
    """Kronecker product of two arrays."""
    return _kronecker_product(a, b)


# --- Signal ---


def convolve(a: Tensor, v: Tensor, mode: str = "full") -> Tensor:
    """Returns the discrete, linear convolution of two one-dimensional sequences."""
    mode_map = {"full": ConvolutionMode.FULL, "valid": ConvolutionMode.VALID, "same": ConvolutionMode.SAME}
    return _convolution_1d(a, v, mode=mode_map[mode])


def hanning(M: int) -> Tensor:
    """Return the Hanning window."""
    return _hanning_window(M, symmetric=True)


def hamming(M: int) -> Tensor:
    """Return the Hamming window."""
    return _hamming_window(M, symmetric=True)


def blackman(M: int) -> Tensor:
    """Return the Blackman window."""
    return _blackman_window(M, symmetric=True)


# --- np.linalg sub-namespace ---
class _LinAlg:
    """Namespace object mimicking numpy.linalg."""

    @staticmethod
    def norm(x: Tensor, ord=None, axis=None, keepdims: bool = False) -> Tensor:
        return _norm(x, order=ord, axis=axis, keep_dimensions=keepdims)

    @staticmethod
    def det(a: Tensor) -> Tensor:
        return _determinant(a)

    @staticmethod
    def inv(a: Tensor) -> Tensor:
        return _inverse(a)

    @staticmethod
    def pinv(a: Tensor) -> Tensor:
        return _pseudo_inverse(a)

    @staticmethod
    def solve(a: Tensor, b: Tensor) -> Tensor:
        return _solve(a, b)

    @staticmethod
    def lstsq(a: Tensor, b: Tensor):
        return _least_squares(a, b)

    @staticmethod
    def cond(x: Tensor, p=None) -> Tensor:
        return _condition_number(x, order=p)

    @staticmethod
    def matrix_rank(a: Tensor) -> Tensor:
        return _matrix_rank(a)

    @staticmethod
    def cholesky(a: Tensor) -> Tensor:
        return _cholesky(a)

    @staticmethod
    def qr(a: Tensor):
        return _qr_decomposition(a)

    @staticmethod
    def matrix_power(a: Tensor, n: int) -> Tensor:
        return _matrix_power(a, n)


linalg = _LinAlg()


# --- np.fft sub-namespace ---
class _FFT:
    """Namespace object mimicking numpy.fft."""

    @staticmethod
    def fft(x: Tensor) -> Tensor:
        return _dft(x)

    @staticmethod
    def ifft(x: Tensor) -> Tensor:
        return _idft(x)

    @staticmethod
    def fftfreq(n: int, d: float = 1.0) -> Tensor:
        return _fourier_frequencies(n, sample_spacing=d)


fft = _FFT()
