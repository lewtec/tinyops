"""Tests for numpy 2.x compatibility layer.

Compares tinyops.compat.numpy2 against actual numpy.
"""

import numpy as np
from tinygrad import Tensor

from tinyops._core import assert_close
from tinyops.compat import numpy2 as tnp

# ============================================================================
# Statistics
# ============================================================================


class TestMean:
    def test_global_mean(self):
        data = np.random.randn(50).astype(np.float32)
        assert_close(tnp.mean(Tensor(data)), np.mean(data))

    def test_axis_0(self):
        data = np.random.randn(10, 5).astype(np.float32)
        assert_close(tnp.mean(Tensor(data), axis=0), np.mean(data, axis=0))

    def test_axis_1_keepdims(self):
        data = np.random.randn(4, 6).astype(np.float32)
        result = tnp.mean(Tensor(data), axis=1, keepdims=True)
        expected = np.mean(data, axis=1, keepdims=True)
        assert_close(result, expected)
        assert result.shape == expected.shape

    def test_single_element(self):
        data = np.array([42.0], dtype=np.float32)
        assert_close(tnp.mean(Tensor(data)), np.mean(data))

    def test_3d(self):
        data = np.random.randn(3, 4, 5).astype(np.float32)
        assert_close(tnp.mean(Tensor(data), axis=2), np.mean(data, axis=2))


class TestMedian:
    def test_odd_length(self):
        data = np.array([1, 3, 5, 7, 9], dtype=np.float32)
        assert_close(tnp.median(Tensor(data)), np.median(data))

    def test_even_length(self):
        data = np.array([1, 2, 3, 4], dtype=np.float32)
        assert_close(tnp.median(Tensor(data)), np.median(data))

    def test_random(self):
        data = np.random.randn(101).astype(np.float32)
        assert_close(tnp.median(Tensor(data)), np.median(data))


class TestStd:
    def test_population_std(self):
        data = np.random.randn(100).astype(np.float32)
        assert_close(tnp.std(Tensor(data)), np.std(data))

    def test_sample_std(self):
        data = np.random.randn(100).astype(np.float32)
        assert_close(tnp.std(Tensor(data), ddof=1), np.std(data, ddof=1), atol=1e-4)

    def test_axis(self):
        data = np.random.randn(10, 5).astype(np.float32)
        assert_close(tnp.std(Tensor(data), axis=0), np.std(data, axis=0), atol=1e-4)


class TestVar:
    def test_population_var(self):
        data = np.random.randn(100).astype(np.float32)
        assert_close(tnp.var(Tensor(data)), np.var(data), atol=1e-4)

    def test_sample_var(self):
        data = np.random.randn(50).astype(np.float32)
        assert_close(tnp.var(Tensor(data), ddof=1), np.var(data, ddof=1), atol=1e-4)


class TestAverage:
    def test_simple(self):
        data = np.array([1, 2, 3, 4], dtype=np.float32)
        assert_close(tnp.average(Tensor(data)), np.average(data))

    def test_weighted(self):
        data = np.array([1, 2, 3, 4], dtype=np.float32)
        weights = np.array([4, 3, 2, 1], dtype=np.float32)
        assert_close(
            tnp.average(Tensor(data), weights=Tensor(weights)),
            np.average(data, weights=weights),
        )

    def test_returned(self):
        data = np.array([1, 2, 3, 4], dtype=np.float32)
        weights = np.array([4, 3, 2, 1], dtype=np.float32)
        result_avg, result_wsum = tnp.average(Tensor(data), weights=Tensor(weights), returned=True)
        expected_avg, expected_wsum = np.average(data, weights=weights, returned=True)
        assert_close(result_avg, expected_avg)
        assert_close(result_wsum, expected_wsum)


class TestPercentile:
    def test_50th(self):
        data = np.arange(1, 11, dtype=np.float32)
        assert_close(tnp.percentile(Tensor(data), 50), np.percentile(data, 50), atol=1e-4)

    def test_boundaries(self):
        data = np.arange(1, 101, dtype=np.float32)
        assert_close(tnp.percentile(Tensor(data), 0), np.percentile(data, 0), atol=1e-4)
        assert_close(tnp.percentile(Tensor(data), 100), np.percentile(data, 100), atol=1e-4)


class TestPtp:
    def test_1d(self):
        data = np.array([3, 1, 7, 2], dtype=np.float32)
        assert_close(tnp.ptp(Tensor(data)), np.ptp(data))

    def test_axis(self):
        data = np.random.randn(4, 5).astype(np.float32)
        assert_close(tnp.ptp(Tensor(data), axis=1), np.ptp(data, axis=1))


class TestBincount:
    def test_simple(self):
        data = np.array([0, 1, 1, 2, 3, 3, 3])
        result = tnp.bincount(Tensor(data))
        expected = np.bincount(data)
        assert_close(result, expected.astype(np.float32))

    def test_minlength(self):
        data = np.array([0, 1, 2])
        result = tnp.bincount(Tensor(data), minlength=5)
        expected = np.bincount(data, minlength=5)
        assert_close(result, expected.astype(np.float32))


class TestDigitize:
    def test_basic(self):
        data = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
        bins = np.array([1, 2, 3], dtype=np.float32)
        result = tnp.digitize(Tensor(data), Tensor(bins))
        expected = np.digitize(data, bins)
        assert_close(result, expected.astype(np.float32))


class TestCorrcoef:
    def test_two_variables(self):
        x = np.random.randn(2, 50).astype(np.float32)
        assert_close(tnp.corrcoef(Tensor(x)), np.corrcoef(x), atol=1e-4)


class TestCorrelate:
    def test_valid(self):
        a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        v = np.array([1, 0, -1], dtype=np.float32)
        assert_close(tnp.correlate(Tensor(a), Tensor(v), mode="valid"), np.correlate(a, v, mode="valid"))

    def test_full(self):
        a = np.array([1, 2, 3], dtype=np.float32)
        v = np.array([1, 2], dtype=np.float32)
        assert_close(tnp.correlate(Tensor(a), Tensor(v), mode="full"), np.correlate(a, v, mode="full"))


class TestCov:
    def test_basic(self):
        data = np.random.randn(3, 50).astype(np.float32)
        assert_close(tnp.cov(Tensor(data)), np.cov(data), atol=1e-3)


class TestHistogram:
    def test_basic(self):
        data = np.random.randn(200).astype(np.float32)
        counts, edges = tnp.histogram(Tensor(data), bins=10)
        expected_counts, expected_edges = np.histogram(data, bins=10)
        assert_close(counts, expected_counts.astype(np.float32), atol=1)
        assert_close(edges, expected_edges.astype(np.float32), atol=1e-4)


class TestHistogram2d:
    def test_basic(self):
        x = np.random.randn(200).astype(np.float32)
        y = np.random.randn(200).astype(np.float32)
        counts, xedges, yedges = tnp.histogram2d(Tensor(x), Tensor(y), bins=5)
        expected_counts, expected_xedges, expected_yedges = np.histogram2d(x, y, bins=5)
        assert_close(counts, expected_counts.astype(np.float32), atol=1)


# ============================================================================
# Linear Algebra (top-level)
# ============================================================================


class TestDot:
    def test_1d(self):
        a = np.random.randn(10).astype(np.float32)
        b = np.random.randn(10).astype(np.float32)
        assert_close(tnp.dot(Tensor(a), Tensor(b)), np.dot(a, b), atol=1e-4)

    def test_2d(self):
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(4, 2).astype(np.float32)
        assert_close(tnp.dot(Tensor(a), Tensor(b)), np.dot(a, b), atol=1e-4)


class TestMatmul:
    def test_basic(self):
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(4, 5).astype(np.float32)
        assert_close(tnp.matmul(Tensor(a), Tensor(b)), np.matmul(a, b), atol=1e-4)


class TestVdot:
    def test_basic(self):
        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([4, 5, 6], dtype=np.float32)
        assert_close(tnp.vdot(Tensor(a), Tensor(b)), np.vdot(a, b))


class TestInner:
    def test_1d(self):
        a = np.random.randn(5).astype(np.float32)
        b = np.random.randn(5).astype(np.float32)
        assert_close(tnp.inner(Tensor(a), Tensor(b)), np.inner(a, b), atol=1e-4)


class TestOuter:
    def test_basic(self):
        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([4, 5], dtype=np.float32)
        assert_close(tnp.outer(Tensor(a), Tensor(b)), np.outer(a, b))


class TestTensordot:
    def test_basic(self):
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(4, 5).astype(np.float32)
        assert_close(tnp.tensordot(Tensor(a), Tensor(b), axes=1), np.tensordot(a, b, axes=1), atol=1e-4)


class TestEinsum:
    def test_matrix_multiply(self):
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(4, 5).astype(np.float32)
        assert_close(tnp.einsum("ij,jk->ik", Tensor(a), Tensor(b)), np.einsum("ij,jk->ik", a, b), atol=1e-4)

    def test_trace(self):
        a = np.random.randn(4, 4).astype(np.float32)
        assert_close(tnp.einsum("ii->", Tensor(a)), np.einsum("ii->", a), atol=1e-4)


class TestTrace:
    def test_basic(self):
        a = np.random.randn(4, 4).astype(np.float32)
        assert_close(tnp.trace(Tensor(a)), np.trace(a), atol=1e-4)


class TestDiagonal:
    def test_basic(self):
        a = np.random.randn(4, 4).astype(np.float32)
        assert_close(tnp.diagonal(Tensor(a)), np.diagonal(a), atol=1e-5)

    def test_offset(self):
        a = np.random.randn(5, 5).astype(np.float32)
        assert_close(tnp.diagonal(Tensor(a), offset=1), np.diagonal(a, offset=1), atol=1e-5)


class TestKron:
    def test_basic(self):
        a = np.array([[1, 0], [0, 1]], dtype=np.float32)
        b = np.array([[1, 2], [3, 4]], dtype=np.float32)
        assert_close(tnp.kron(Tensor(a), Tensor(b)), np.kron(a, b))


# ============================================================================
# np.linalg
# ============================================================================


class TestLinalgNorm:
    def test_vector_l2(self):
        data = np.random.randn(10).astype(np.float32)
        assert_close(tnp.linalg.norm(Tensor(data)), np.linalg.norm(data), atol=1e-4)

    def test_vector_l1(self):
        data = np.random.randn(10).astype(np.float32)
        assert_close(tnp.linalg.norm(Tensor(data), ord=1), np.linalg.norm(data, ord=1), atol=1e-4)

    def test_frobenius(self):
        data = np.random.randn(4, 4).astype(np.float32)
        assert_close(tnp.linalg.norm(Tensor(data), ord="fro"), np.linalg.norm(data, ord="fro"), atol=1e-4)

    def test_inf_norm(self):
        data = np.random.randn(3, 3).astype(np.float32)
        assert_close(tnp.linalg.norm(Tensor(data), ord=float("inf")), np.linalg.norm(data, ord=np.inf), atol=1e-4)


class TestLinalgDet:
    def test_basic(self):
        a = np.random.randn(3, 3).astype(np.float32)
        assert_close(tnp.linalg.det(Tensor(a)), np.linalg.det(a), atol=1e-3)

    def test_identity(self):
        a = np.eye(4, dtype=np.float32)
        assert_close(tnp.linalg.det(Tensor(a)), np.linalg.det(a), atol=1e-5)


class TestLinalgInv:
    def test_basic(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        assert_close(tnp.linalg.inv(Tensor(a)), np.linalg.inv(a), atol=1e-4)

    def test_identity(self):
        a = np.eye(3, dtype=np.float32)
        assert_close(tnp.linalg.inv(Tensor(a)), np.linalg.inv(a), atol=1e-5)


class TestLinalgSolve:
    def test_basic(self):
        a = np.array([[3, 1], [1, 2]], dtype=np.float32)
        b = np.array([9, 8], dtype=np.float32)
        assert_close(tnp.linalg.solve(Tensor(a), Tensor(b)), np.linalg.solve(a, b), atol=1e-4)


class TestLinalgCholesky:
    def test_basic(self):
        a = np.array([[4, 2], [2, 3]], dtype=np.float32)
        assert_close(tnp.linalg.cholesky(Tensor(a)), np.linalg.cholesky(a), atol=1e-4)


class TestLinalgQr:
    def test_basic(self):
        a = np.random.randn(4, 3).astype(np.float32)
        q_tn, r_tn = tnp.linalg.qr(Tensor(a))
        q_np, r_np = np.linalg.qr(a)
        # QR can differ by sign; check Q @ R = A
        assert_close(q_tn.matmul(r_tn), Tensor(a), atol=1e-4)


class TestLinalgMatrixPower:
    def test_square(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        assert_close(tnp.linalg.matrix_power(Tensor(a), 3), np.linalg.matrix_power(a, 3), atol=1e-2)


class TestLinalgPinv:
    def test_basic(self):
        a = np.random.randn(4, 3).astype(np.float32)
        assert_close(tnp.linalg.pinv(Tensor(a)), np.linalg.pinv(a), atol=1e-3)


class TestLinalgMatrixRank:
    def test_full_rank(self):
        a = np.eye(3, dtype=np.float32)
        assert_close(tnp.linalg.matrix_rank(Tensor(a)), np.linalg.matrix_rank(a))


class TestLinalgLstsq:
    def test_basic(self):
        a = np.array([[1, 1], [1, 2], [1, 3]], dtype=np.float32)
        b = np.array([1, 2, 3], dtype=np.float32)
        result = tnp.linalg.lstsq(Tensor(a), Tensor(b))
        expected = np.linalg.lstsq(a, b, rcond=None)[0]
        assert_close(result, expected, atol=1e-3)


# ============================================================================
# Signal / Windows
# ============================================================================


class TestConvolve:
    def test_full(self):
        a = np.array([1, 2, 3], dtype=np.float32)
        v = np.array([0, 1, 0.5], dtype=np.float32)
        assert_close(tnp.convolve(Tensor(a), Tensor(v), mode="full"), np.convolve(a, v, mode="full"), atol=1e-5)

    def test_valid(self):
        a = np.random.randn(20).astype(np.float32)
        v = np.random.randn(5).astype(np.float32)
        assert_close(tnp.convolve(Tensor(a), Tensor(v), mode="valid"), np.convolve(a, v, mode="valid"), atol=1e-4)

    def test_same(self):
        a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        v = np.array([1, 1, 1], dtype=np.float32)
        assert_close(tnp.convolve(Tensor(a), Tensor(v), mode="same"), np.convolve(a, v, mode="same"), atol=1e-5)


class TestHanning:
    def test_basic(self):
        assert_close(tnp.hanning(10), np.hanning(10).astype(np.float32), atol=1e-5)

    def test_single(self):
        assert_close(tnp.hanning(1), np.hanning(1).astype(np.float32), atol=1e-5)


class TestHamming:
    def test_basic(self):
        assert_close(tnp.hamming(10), np.hamming(10).astype(np.float32), atol=1e-5)


class TestBlackman:
    def test_basic(self):
        assert_close(tnp.blackman(10), np.blackman(10).astype(np.float32), atol=1e-5)


# ============================================================================
# np.fft
# ============================================================================


class TestFFT:
    def test_basic(self):
        data = np.random.randn(16).astype(np.float32)
        # Our DFT expects (N, 2) shaped input: columns are [real, imag]
        complex_input = Tensor(np.stack([data, np.zeros_like(data)], axis=1))
        result = tnp.fft.fft(complex_input)
        expected = np.fft.fft(data)
        assert_close(result[:, 0], expected.real.astype(np.float32), atol=1e-3)
        assert_close(result[:, 1], expected.imag.astype(np.float32), atol=1e-3)


class TestIFFT:
    def test_roundtrip(self):
        data = np.random.randn(16).astype(np.float32)
        complex_input = Tensor(np.stack([data, np.zeros_like(data)], axis=1))
        freq = tnp.fft.fft(complex_input)
        recovered = tnp.fft.ifft(freq)
        assert_close(recovered[:, 0], data, atol=1e-3)


class TestFFTFreq:
    def test_basic(self):
        result = tnp.fft.fftfreq(8, d=1.0)
        expected = np.fft.fftfreq(8, d=1.0).astype(np.float32)
        assert_close(result, expected, atol=1e-5)

    def test_custom_spacing(self):
        result = tnp.fft.fftfreq(10, d=0.5)
        expected = np.fft.fftfreq(10, d=0.5).astype(np.float32)
        assert_close(result, expected, atol=1e-5)
