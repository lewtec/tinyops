"""Tests for scikit-learn compatibility layer.

Compares tinyops.compat.sklearn against actual sklearn.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import (
    Binarizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)
from tinygrad import Tensor

from tinyops._core import assert_close
from tinyops.compat import sklearn as tsk

# ============================================================================
# Preprocessing
# ============================================================================


class TestStandardScaler:
    def test_basic(self):
        data = np.random.randn(50, 3).astype(np.float32)
        result = tsk.preprocessing.StandardScaler().fit_transform(Tensor(data))
        expected = StandardScaler().fit_transform(data)
        assert_close(result, expected.astype(np.float32), atol=1e-4)

    def test_single_feature(self):
        data = np.random.randn(30, 1).astype(np.float32)
        result = tsk.preprocessing.StandardScaler().fit_transform(Tensor(data))
        expected = StandardScaler().fit_transform(data)
        assert_close(result, expected.astype(np.float32), atol=1e-4)

    def test_zero_variance(self):
        data = np.ones((10, 2), dtype=np.float32)
        result = tsk.preprocessing.StandardScaler().fit_transform(Tensor(data))
        # Zero variance columns should remain constant (zeroed out by sklearn)
        assert result.shape == (10, 2)


class TestMinMaxScaler:
    def test_default_range(self):
        data = np.random.randn(50, 3).astype(np.float32)
        result = tsk.preprocessing.MinMaxScaler().fit_transform(Tensor(data))
        expected = MinMaxScaler().fit_transform(data)
        assert_close(result, expected.astype(np.float32), atol=1e-4)

    def test_custom_range(self):
        data = np.random.randn(50, 3).astype(np.float32)
        result = tsk.preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(Tensor(data))
        expected = MinMaxScaler(feature_range=(-1, 1)).fit_transform(data)
        assert_close(result, expected.astype(np.float32), atol=1e-4)


class TestMaxAbsScaler:
    def test_basic(self):
        data = np.random.randn(50, 3).astype(np.float32)
        result = tsk.preprocessing.MaxAbsScaler().fit_transform(Tensor(data))
        expected = MaxAbsScaler().fit_transform(data)
        assert_close(result, expected.astype(np.float32), atol=1e-4)


class TestRobustScaler:
    def test_basic(self):
        data = np.random.randn(100, 3).astype(np.float32)
        result = tsk.preprocessing.RobustScaler().fit_transform(Tensor(data))
        expected = RobustScaler().fit_transform(data)
        assert_close(result, expected.astype(np.float32), atol=1e-2)


class TestNormalizer:
    def test_l2(self):
        data = np.random.randn(10, 4).astype(np.float32)
        result = tsk.preprocessing.Normalizer(norm="l2").fit_transform(Tensor(data))
        expected = Normalizer(norm="l2").fit_transform(data)
        assert_close(result, expected.astype(np.float32), atol=1e-4)

    def test_l1(self):
        data = np.abs(np.random.randn(10, 4).astype(np.float32)) + 0.01
        result = tsk.preprocessing.Normalizer(norm="l1").fit_transform(Tensor(data))
        expected = Normalizer(norm="l1").fit_transform(data)
        assert_close(result, expected.astype(np.float32), atol=1e-4)

    def test_max(self):
        data = np.random.randn(10, 4).astype(np.float32)
        result = tsk.preprocessing.Normalizer(norm="max").fit_transform(Tensor(data))
        expected = Normalizer(norm="max").fit_transform(data)
        assert_close(result, expected.astype(np.float32), atol=1e-4)


class TestBinarizer:
    def test_default(self):
        data = np.array([[-1, 0.5, 2], [0, -0.5, 1]], dtype=np.float32)
        result = tsk.preprocessing.Binarizer().fit_transform(Tensor(data))
        expected = Binarizer().fit_transform(data)
        assert_close(result, expected.astype(np.float32))

    def test_custom_threshold(self):
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result = tsk.preprocessing.Binarizer(threshold=3.0).fit_transform(Tensor(data))
        expected = Binarizer(threshold=3.0).fit_transform(data)
        assert_close(result, expected.astype(np.float32))


class TestPolynomialFeatures:
    def test_degree2(self):
        data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        result = tsk.preprocessing.PolynomialFeatures(degree=2).fit_transform(Tensor(data))
        expected = PolynomialFeatures(degree=2).fit_transform(data)
        assert_close(result, expected.astype(np.float32))

    def test_interaction_only(self):
        data = np.array([[1, 2, 3]], dtype=np.float32)
        result = tsk.preprocessing.PolynomialFeatures(degree=2, interaction_only=True).fit_transform(Tensor(data))
        expected = PolynomialFeatures(degree=2, interaction_only=True).fit_transform(data)
        assert_close(result, expected.astype(np.float32))

    def test_no_bias(self):
        data = np.array([[1, 2]], dtype=np.float32)
        result = tsk.preprocessing.PolynomialFeatures(degree=2, include_bias=False).fit_transform(Tensor(data))
        expected = PolynomialFeatures(degree=2, include_bias=False).fit_transform(data)
        assert_close(result, expected.astype(np.float32))


# ============================================================================
# Metrics
# ============================================================================


class TestAccuracyScore:
    def test_perfect(self):
        y_true = Tensor([0, 1, 1, 0])
        y_pred = Tensor([0, 1, 1, 0])
        result = tsk.metrics.accuracy_score(y_true, y_pred)
        assert_close(result, np.float32(1.0))

    def test_half_correct(self):
        y_true = Tensor([0, 1, 0, 1])
        y_pred = Tensor([1, 1, 1, 1])
        result = tsk.metrics.accuracy_score(y_true, y_pred)
        assert_close(result, np.float32(0.5))


class TestPrecisionScore:
    def test_basic(self):
        y_true = np.array([0, 1, 1, 0, 1, 0], dtype=np.float32)
        y_pred = np.array([0, 1, 0, 0, 1, 1], dtype=np.float32)
        result = tsk.metrics.precision_score(Tensor(y_true), Tensor(y_pred))
        expected = precision_score(y_true, y_pred)
        assert_close(result, np.float32(expected), atol=1e-4)


class TestRecallScore:
    def test_basic(self):
        y_true = np.array([0, 1, 1, 0, 1, 0], dtype=np.float32)
        y_pred = np.array([0, 1, 0, 0, 1, 1], dtype=np.float32)
        result = tsk.metrics.recall_score(Tensor(y_true), Tensor(y_pred))
        expected = recall_score(y_true, y_pred)
        assert_close(result, np.float32(expected), atol=1e-4)


class TestF1Score:
    def test_basic(self):
        y_true = np.array([0, 1, 1, 0, 1, 0], dtype=np.float32)
        y_pred = np.array([0, 1, 0, 0, 1, 1], dtype=np.float32)
        result = tsk.metrics.f1_score(Tensor(y_true), Tensor(y_pred))
        expected = f1_score(y_true, y_pred)
        assert_close(result, np.float32(expected), atol=1e-4)


class TestConfusionMatrix:
    def test_binary(self):
        y_true = np.array([0, 1, 0, 1, 1, 0], dtype=np.float32)
        y_pred = np.array([0, 1, 1, 1, 0, 0], dtype=np.float32)
        result = tsk.metrics.confusion_matrix(Tensor(y_true), Tensor(y_pred))
        expected = confusion_matrix(y_true, y_pred)
        assert_close(result, expected.astype(np.float32))


class TestRocAucScore:
    def test_basic(self):
        y_true = np.array([0, 0, 1, 1], dtype=np.float32)
        y_score = np.array([0.1, 0.4, 0.35, 0.8], dtype=np.float32)
        result = tsk.metrics.roc_auc_score(Tensor(y_true), Tensor(y_score))
        expected = roc_auc_score(y_true, y_score)
        assert_close(result, np.float32(expected), atol=1e-3)


class TestMSE:
    def test_basic(self):
        y_true = np.array([3, -0.5, 2, 7], dtype=np.float32)
        y_pred = np.array([2.5, 0.0, 2, 8], dtype=np.float32)
        result = tsk.metrics.mean_squared_error(Tensor(y_true), Tensor(y_pred))
        expected = mean_squared_error(y_true, y_pred)
        assert_close(result, np.float32(expected), atol=1e-4)


class TestMAE:
    def test_basic(self):
        y_true = np.array([3, -0.5, 2, 7], dtype=np.float32)
        y_pred = np.array([2.5, 0.0, 2, 8], dtype=np.float32)
        result = tsk.metrics.mean_absolute_error(Tensor(y_true), Tensor(y_pred))
        expected = mean_absolute_error(y_true, y_pred)
        assert_close(result, np.float32(expected), atol=1e-4)


class TestR2Score:
    def test_basic(self):
        y_true = np.array([3, -0.5, 2, 7], dtype=np.float32)
        y_pred = np.array([2.5, 0.0, 2, 8], dtype=np.float32)
        result = tsk.metrics.r2_score(Tensor(y_true), Tensor(y_pred))
        expected = r2_score(y_true, y_pred)
        assert_close(result, np.float32(expected), atol=1e-3)


class TestPairwiseDistances:
    def test_hamming(self):
        data = np.array([[1, 0, 1], [0, 0, 1], [1, 1, 0]], dtype=np.float32)
        result = tsk.metrics.pairwise.pairwise_distances(Tensor(data), metric="hamming")
        expected = pairwise_distances(data, metric="hamming")
        assert_close(result, expected.astype(np.float32), atol=1e-4)


# ============================================================================
# Feature extraction
# ============================================================================


class TestCountVectorizer:
    def test_basic(self):
        corpus = ["hello world", "world hello hello"]
        result = tsk.feature_extraction.text.CountVectorizer().fit_transform(corpus)
        expected = CountVectorizer().fit_transform(corpus).toarray()
        assert_close(result, expected.astype(np.float32))


class TestTfidfVectorizer:
    def test_basic(self):
        corpus = ["the cat sat", "the cat played", "the dog sat"]
        result = tsk.feature_extraction.text.TfidfVectorizer().fit_transform(corpus)
        expected = TfidfVectorizer().fit_transform(corpus).toarray()
        assert_close(result, expected.astype(np.float32), atol=1e-4)