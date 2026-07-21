"""Tests for numpy 2.x compatibility layer.

Compares tinyops.compat.numpy2 against actual numpy.
"""

import numpy as np
import pytest
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

    def test_axis(self):
        data = np.random.randn(4, 5).astype(np.float32)
        assert_close(
            tnp.percentile(Tensor(data), 25, axis=0),
            np.percentile(data, 25, axis=0),
            atol=1e-4,
        )
        assert_close(
            tnp.percentile(Tensor(data), 75, axis=1),
            np.percentile(data, 75, axis=1),
            atol=1e-4,
        )

    def test_keepdims(self):
        data = np.random.randn(3, 6).astype(np.float32)
        result = tnp.percentile(Tensor(data), 50, axis=1, keepdims=True)
        expected = np.percentile(data, 50, axis=1, keepdims=True)
        assert_close(result, expected, atol=1e-4)
        assert result.shape == expected.shape

    def test_multiple_percentiles(self):
        data = np.random.randn(5, 4).astype(np.float32)
        qs = [10, 50, 90]
        result = tnp.percentile(Tensor(data), qs, axis=0)
        expected = np.percentile(data, qs, axis=0)
        assert_close(result, expected.astype(np.float32), atol=1e-4)
        assert result.shape == expected.shape

    def test_negative_axis(self):
        data = np.random.randn(3, 4, 5).astype(np.float32)
        assert_close(
            tnp.percentile(Tensor(data), 40, axis=-1),
            np.percentile(data, 40, axis=-1),
            atol=1e-4,
        )

    def test_unsupported_method(self):
        data = np.arange(1, 6, dtype=np.float32)
        with pytest.raises(NotImplementedError):
            tnp.percentile(Tensor(data), 50, method="nearest")


class TestQuantile:
    def test_median(self):
        data = np.arange(1, 11, dtype=np.float32)
        assert_close(tnp.quantile(Tensor(data), 0.5), np.quantile(data, 0.5), atol=1e-4)

    def test_boundaries(self):
        data = np.arange(1, 51, dtype=np.float32)
        assert_close(tnp.quantile(Tensor(data), 0.0), np.quantile(data, 0.0), atol=1e-4)
        assert_close(tnp.quantile(Tensor(data), 1.0), np.quantile(data, 1.0), atol=1e-4)

    def test_axis_and_multiple(self):
        data = np.random.randn(4, 5).astype(np.float32)
        qs = [0.25, 0.5, 0.75]
        result = tnp.quantile(Tensor(data), qs, axis=0)
        expected = np.quantile(data, qs, axis=0)
        assert_close(result, expected.astype(np.float32), atol=1e-4)
        assert result.shape == expected.shape

    def test_keepdims(self):
        data = np.random.randn(3, 6).astype(np.float32)
        result = tnp.quantile(Tensor(data), 0.5, axis=0, keepdims=True)
        expected = np.quantile(data, 0.5, axis=0, keepdims=True)
        assert_close(result, expected, atol=1e-4)
        assert result.shape == expected.shape

    def test_matches_percentile(self):
        data = np.random.randn(20).astype(np.float32)
        assert_close(
            tnp.quantile(Tensor(data), 0.3),
            tnp.percentile(Tensor(data), 30),
            atol=1e-5,
        )

    def test_unsupported_method(self):
        data = np.arange(1, 6, dtype=np.float32)
        with pytest.raises(NotImplementedError):
            tnp.quantile(Tensor(data), 0.5, method="higher")


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


class TestHistogramdd:
    def test_basic_3d(self):
        rng = np.random.RandomState(0)
        sample = rng.randn(200, 3).astype(np.float32)
        counts, edges = tnp.histogramdd(Tensor(sample), bins=5)
        expected_counts, expected_edges = np.histogramdd(sample, bins=5)
        assert_close(counts, expected_counts.astype(np.float32), atol=1)
        assert len(edges) == 3
        for edge, expected_edge in zip(edges, expected_edges, strict=True):
            assert_close(edge, expected_edge.astype(np.float32), atol=1e-4)

    def test_per_axis_bins_and_range(self):
        rng = np.random.RandomState(1)
        sample = rng.randn(150, 3).astype(np.float32)
        bins = [3, 4, 5]
        value_range = [(-2.0, 2.0), (-1.5, 1.5), (-2.5, 2.5)]
        counts, edges = tnp.histogramdd(Tensor(sample), bins=bins, range=value_range)
        expected_counts, expected_edges = np.histogramdd(sample, bins=bins, range=value_range)
        assert counts.shape == tuple(bins)
        assert_close(counts, expected_counts.astype(np.float32), atol=1)
        for edge, expected_edge in zip(edges, expected_edges, strict=True):
            assert_close(edge, expected_edge.astype(np.float32), atol=1e-4)

    def test_density(self):
        rng = np.random.RandomState(2)
        sample = rng.randn(100, 2).astype(np.float32)
        counts, _ = tnp.histogramdd(Tensor(sample), bins=4, density=True)
        expected_counts, _ = np.histogramdd(sample, bins=4, density=True)
        assert_close(counts, expected_counts.astype(np.float32), atol=1e-3)

    def test_weights(self):
        rng = np.random.RandomState(3)
        sample = rng.randn(80, 2).astype(np.float32)
        weights = rng.rand(80).astype(np.float32)
        counts, _ = tnp.histogramdd(Tensor(sample), bins=4, weights=Tensor(weights))
        expected_counts, _ = np.histogramdd(sample, bins=4, weights=weights)
        assert_close(counts, expected_counts.astype(np.float32), atol=1e-3)

    def test_empty_samples(self):
        sample = np.zeros((0, 2), dtype=np.float32)
        counts, edges = tnp.histogramdd(Tensor(sample), bins=3)
        expected_counts, expected_edges = np.histogramdd(sample, bins=3)
        assert_close(counts, expected_counts.astype(np.float32), atol=0)
        assert counts.shape == (3, 3)
        for edge, expected_edge in zip(edges, expected_edges, strict=True):
            assert_close(edge, expected_edge.astype(np.float32), atol=1e-4)

    def test_one_dimensional_samples(self):
        sample = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
        counts, edges = tnp.histogramdd(Tensor(sample), bins=2, range=[(0.0, 2.0)])
        expected_counts, expected_edges = np.histogramdd(sample, bins=2, range=[(0.0, 2.0)])
        assert_close(counts, expected_counts.astype(np.float32), atol=0)
        assert_close(edges[0], expected_edges[0].astype(np.float32), atol=1e-4)

    def test_matches_histogram2d_layout(self):
        rng = np.random.RandomState(4)
        sample = rng.randn(120, 2).astype(np.float32)
        counts_dd, edges_dd = tnp.histogramdd(Tensor(sample), bins=[5, 6])
        counts_2d, x_edges, y_edges = tnp.histogram2d(
            Tensor(sample[:, 0]), Tensor(sample[:, 1]), bins=[5, 6]
        )
        assert_close(counts_dd, counts_2d, atol=0)
        assert_close(edges_dd[0], x_edges, atol=1e-5)
        assert_close(edges_dd[1], y_edges, atol=1e-5)


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


class TestKaiser:
    def test_basic(self):
        assert_close(tnp.kaiser(10, 14), np.kaiser(10, 14).astype(np.float32), atol=1e-5)

    def test_single(self):
        assert_close(tnp.kaiser(1, 5), np.kaiser(1, 5).astype(np.float32), atol=1e-5)

    def test_empty(self):
        assert tnp.kaiser(0, 5).shape == (0,)
        assert np.kaiser(0, 5).shape == (0,)

    def test_beta_zero_is_rectangular(self):
        assert_close(tnp.kaiser(8, 0), np.kaiser(8, 0).astype(np.float32), atol=1e-5)

    def test_odd_length_peak_one(self):
        expected = np.kaiser(11, 8.6).astype(np.float32)
        assert_close(tnp.kaiser(11, 8.6), expected, atol=1e-5)
        # Center sample is 1 for odd length (numpy contract).
        assert abs(float(expected[5]) - 1.0) < 1e-5

    def test_various_beta(self):
        for length, beta in [(5, 0.5), (12, 5), (16, 8.6), (32, 14), (7, 20)]:
            assert_close(
                tnp.kaiser(length, beta),
                np.kaiser(length, beta).astype(np.float32),
                atol=1e-5,
            )


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


class TestFFT2:
    def test_matches_numpy_small(self):
        # Tiny sizes keep clang kernel compile time under the host memory budget.
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        packed = Tensor(np.stack([data, np.zeros_like(data)], axis=-1))
        result = tnp.fft.fft2(packed)
        expected = np.fft.fft2(data)
        assert result.shape == (2, 3, 2)
        assert_close(result[:, :, 0], expected.real.astype(np.float32), atol=1e-3)
        assert_close(result[:, :, 1], expected.imag.astype(np.float32), atol=1e-3)

    def test_power_of_two_square(self):
        data = np.random.randn(4, 2).astype(np.float32)
        packed = Tensor(np.stack([data, np.zeros_like(data)], axis=-1))
        result = tnp.fft.fft2(packed)
        expected = np.fft.fft2(data)
        assert_close(result[:, :, 0], expected.real.astype(np.float32), atol=1e-3)
        assert_close(result[:, :, 1], expected.imag.astype(np.float32), atol=1e-3)


class TestIFFT2:
    def test_roundtrip(self):
        data = np.random.randn(2, 3).astype(np.float32)
        packed = Tensor(np.stack([data, np.zeros_like(data)], axis=-1))
        spectrum = tnp.fft.fft2(packed)
        recovered = tnp.fft.ifft2(spectrum)
        assert_close(recovered[:, :, 0], data, atol=1e-3)
        assert_close(recovered[:, :, 1], np.zeros_like(data), atol=1e-3)

    def test_matches_numpy(self):
        data = np.random.randn(3, 2).astype(np.float32) + 1j * np.random.randn(3, 2).astype(np.float32)
        packed = Tensor(np.stack([data.real.astype(np.float32), data.imag.astype(np.float32)], axis=-1))
        result = tnp.fft.ifft2(packed)
        expected = np.fft.ifft2(data)
        assert_close(result[:, :, 0], expected.real.astype(np.float32), atol=1e-3)
        assert_close(result[:, :, 1], expected.imag.astype(np.float32), atol=1e-3)


class TestFFTFreq:
    def test_basic(self):
        result = tnp.fft.fftfreq(8, d=1.0)
        expected = np.fft.fftfreq(8, d=1.0).astype(np.float32)
        assert_close(result, expected, atol=1e-5)

    def test_custom_spacing(self):
        result = tnp.fft.fftfreq(10, d=0.5)
        expected = np.fft.fftfreq(10, d=0.5).astype(np.float32)
        assert_close(result, expected, atol=1e-5)


class TestRFFT:
    def test_even_length(self):
        data = np.random.randn(8).astype(np.float32)
        result = tnp.fft.rfft(Tensor(data))
        expected = np.fft.rfft(data)
        assert result.shape == (5, 2)
        assert_close(result[:, 0], expected.real.astype(np.float32), atol=1e-3)
        assert_close(result[:, 1], expected.imag.astype(np.float32), atol=1e-3)

    def test_odd_length(self):
        data = np.random.randn(7).astype(np.float32)
        result = tnp.fft.rfft(Tensor(data))
        expected = np.fft.rfft(data)
        assert result.shape == (4, 2)
        assert_close(result[:, 0], expected.real.astype(np.float32), atol=1e-3)
        assert_close(result[:, 1], expected.imag.astype(np.float32), atol=1e-3)

    def test_matches_fft_prefix(self):
        data = np.random.randn(4).astype(np.float32)
        complex_input = Tensor(np.stack([data, np.zeros_like(data)], axis=1))
        full = tnp.fft.fft(complex_input)
        half = tnp.fft.rfft(Tensor(data))
        assert_close(half, full[:3], atol=1e-5)

    def test_single_and_two_samples(self):
        for length in (1, 2):
            data = np.random.randn(length).astype(np.float32)
            result = tnp.fft.rfft(Tensor(data))
            expected = np.fft.rfft(data)
            assert_close(result[:, 0], expected.real.astype(np.float32), atol=1e-4)
            assert_close(result[:, 1], expected.imag.astype(np.float32), atol=1e-4)


class TestIRFFT:
    def test_roundtrip_even(self):
        data = np.random.randn(8).astype(np.float32)
        spectrum = tnp.fft.rfft(Tensor(data))
        recovered = tnp.fft.irfft(spectrum)
        assert_close(recovered, data, atol=1e-3)

    def test_roundtrip_odd_with_n(self):
        data = np.random.randn(7).astype(np.float32)
        spectrum = tnp.fft.rfft(Tensor(data))
        recovered = tnp.fft.irfft(spectrum, n=7)
        assert_close(recovered, data, atol=1e-3)

    def test_matches_numpy(self):
        data = np.random.randn(6).astype(np.float32)
        spectrum = np.fft.rfft(data)
        packed = Tensor(np.stack([spectrum.real, spectrum.imag], axis=1).astype(np.float32))
        result = tnp.fft.irfft(packed, n=6)
        expected = np.fft.irfft(spectrum, n=6).astype(np.float32)
        assert_close(result, expected, atol=1e-3)

    def test_default_n_is_even(self):
        spectrum = np.fft.rfft(np.array([1, 2, 3, 4], dtype=np.float32))
        packed = Tensor(np.stack([spectrum.real, spectrum.imag], axis=1).astype(np.float32))
        result = tnp.fft.irfft(packed)
        expected = np.fft.irfft(spectrum).astype(np.float32)
        assert result.shape == (4,)
        assert_close(result, expected, atol=1e-3)
