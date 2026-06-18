"""scikit-learn compatibility layer.

Provides sklearn-compatible class/function signatures that delegate to tinyops.ops.
Organized into sub-namespaces matching sklearn's package structure.
"""

from tinygrad import Tensor

from tinyops.ops.machine_learning.accuracy_score import accuracy_score as _accuracy_score
from tinyops.ops.machine_learning.bernoulli_naive_bayes import bernoulli_naive_bayes as _bnb
from tinyops.ops.machine_learning.binarizer import binarizer as _binarizer
from tinyops.ops.machine_learning.coefficient_of_determination import coefficient_of_determination as _r2
from tinyops.ops.machine_learning.confusion_matrix import confusion_matrix as _confusion_matrix
from tinyops.ops.machine_learning.f1_score import f1_score as _f1_score
from tinyops.ops.machine_learning.label_encoder import label_encoder as _label_encoder
from tinyops.ops.machine_learning.max_absolute_scaler import max_absolute_scaler as _max_absolute_scaler
from tinyops.ops.machine_learning.mean_absolute_error import mean_absolute_error as _mae
from tinyops.ops.machine_learning.mean_squared_error import mean_squared_error as _mse
from tinyops.ops.machine_learning.min_max_scaler import min_max_scaler as _min_max_scaler
from tinyops.ops.machine_learning.multinomial_naive_bayes import multinomial_naive_bayes as _mnb
from tinyops.ops.machine_learning.nearest_neighbors import nearest_neighbors as _nearest_neighbors
from tinyops.ops.machine_learning.non_negative_matrix_factorization import (
    non_negative_matrix_factorization as _nmf,
)
from tinyops.ops.machine_learning.normalizer import NormType
from tinyops.ops.machine_learning.normalizer import normalizer as _normalizer
from tinyops.ops.machine_learning.one_hot_encoder import one_hot_encoder as _one_hot_encoder
from tinyops.ops.machine_learning.polynomial_features import polynomial_features as _polynomial_features
from tinyops.ops.machine_learning.precision_score import precision_score as _precision_score
from tinyops.ops.machine_learning.recall_score import recall_score as _recall_score
from tinyops.ops.machine_learning.receiver_operating_characteristic import (
    receiver_operating_characteristic_area as _roc_auc,
)
from tinyops.ops.machine_learning.robust_scaler import robust_scaler as _robust_scaler

# --- ops imports ---
from tinyops.ops.machine_learning.standard_scaler import standard_scaler as _standard_scaler
from tinyops.ops.text.count_vectorizer import count_vectorizer as _count_vectorizer
from tinyops.ops.text.pairwise_hamming_distance import pairwise_hamming_distance as _pairwise_hamming
from tinyops.ops.text.tfidf_vectorizer import tfidf_vectorizer as _tfidf_vectorizer

# ============================================================================
# preprocessing
# ============================================================================


class _Preprocessing:
    """Namespace mimicking sklearn.preprocessing."""

    class StandardScaler:
        """Standardize features by removing the mean and scaling to unit variance."""

        def fit_transform(self, X: Tensor) -> Tensor:
            return _standard_scaler(X)

    class MinMaxScaler:
        """Scale features to a given range."""

        def __init__(self, feature_range: tuple[float, float] = (0, 1)):
            self.feature_range = feature_range

        def fit_transform(self, X: Tensor) -> Tensor:
            return _min_max_scaler(X, target_range=self.feature_range)

    class MaxAbsScaler:
        """Scale each feature by its maximum absolute value."""

        def fit_transform(self, X: Tensor) -> Tensor:
            return _max_absolute_scaler(X)

    class RobustScaler:
        """Scale features using statistics that are robust to outliers."""

        def fit_transform(self, X: Tensor) -> Tensor:
            return _robust_scaler(X)

    class Normalizer:
        """Normalize samples individually to unit norm."""

        _NORM_MAP = {"l1": NormType.L1, "l2": NormType.L2, "max": NormType.MAX}

        def __init__(self, norm: str = "l2"):
            self.norm = norm

        def fit_transform(self, X: Tensor) -> Tensor:
            return _normalizer(X, norm_type=self._NORM_MAP[self.norm])

    class Binarizer:
        """Binarize data (set feature values to 0 or 1) according to a threshold."""

        def __init__(self, threshold: float = 0.0):
            self.threshold = threshold

        def fit_transform(self, X: Tensor) -> Tensor:
            return _binarizer(X, threshold=self.threshold)

    class OneHotEncoder:
        """Encode categorical features as a one-hot numeric array."""

        def fit_transform(self, X: Tensor) -> Tensor:
            return _one_hot_encoder(X)

    class LabelEncoder:
        """Encode target labels with value between 0 and n_classes-1."""

        def fit_transform(self, y: Tensor) -> Tensor:
            return _label_encoder(y)

    class PolynomialFeatures:
        """Generate polynomial and interaction features."""

        def __init__(
            self, degree: int = 2, interaction_only: bool = False, include_bias: bool = True
        ):
            self.degree = degree
            self.interaction_only = interaction_only
            self.include_bias = include_bias

        def fit_transform(self, X: Tensor) -> Tensor:
            return _polynomial_features(
                X,
                degree=self.degree,
                interaction_only=self.interaction_only,
                include_bias=self.include_bias,
            )


# ============================================================================
# metrics
# ============================================================================


class _Metrics:
    """Namespace mimicking sklearn.metrics."""

    @staticmethod
    def accuracy_score(y_true: Tensor, y_pred: Tensor) -> Tensor:
        return _accuracy_score(y_true, y_pred)

    @staticmethod
    def precision_score(y_true: Tensor, y_pred: Tensor) -> Tensor:
        return _precision_score(y_true, y_pred)

    @staticmethod
    def recall_score(y_true: Tensor, y_pred: Tensor) -> Tensor:
        return _recall_score(y_true, y_pred)

    @staticmethod
    def f1_score(y_true: Tensor, y_pred: Tensor) -> Tensor:
        return _f1_score(y_true, y_pred)

    @staticmethod
    def confusion_matrix(y_true: Tensor, y_pred: Tensor, labels: list[int] | None = None) -> Tensor:
        return _confusion_matrix(y_true, y_pred, label_values=labels)

    @staticmethod
    def roc_auc_score(y_true: Tensor, y_score: Tensor) -> Tensor:
        return _roc_auc(y_true, y_score)

    @staticmethod
    def mean_squared_error(y_true: Tensor, y_pred: Tensor) -> Tensor:
        return _mse(y_true, y_pred)

    @staticmethod
    def mean_absolute_error(y_true: Tensor, y_pred: Tensor) -> Tensor:
        return _mae(y_true, y_pred)

    @staticmethod
    def r2_score(y_true: Tensor, y_pred: Tensor) -> Tensor:
        return _r2(y_true, y_pred)

    class _Pairwise:
        """Namespace mimicking sklearn.metrics.pairwise."""

        @staticmethod
        def pairwise_distances(X: Tensor, metric: str = "hamming") -> Tensor:
            if metric != "hamming":
                raise NotImplementedError(f"Only 'hamming' metric is supported, got '{metric}'")
            return _pairwise_hamming(X)

    pairwise = _Pairwise()


# ============================================================================
# neighbors
# ============================================================================


class _Neighbors:
    """Namespace mimicking sklearn.neighbors."""

    class NearestNeighbors:
        """Unsupervised learner for implementing neighbor searches."""

        def __init__(self, n_neighbors: int = 5):
            self.n_neighbors = n_neighbors
            self._X = None

        def fit(self, X: Tensor) -> "NearestNeighbors":
            self._X = X
            return self

        def kneighbors(self, X: Tensor | None = None) -> Tensor:
            if X is None:
                X = self._X
            return _nearest_neighbors(X, neighbor_count=self.n_neighbors)


# ============================================================================
# naive_bayes
# ============================================================================


class _NaiveBayes:
    """Namespace mimicking sklearn.naive_bayes."""

    class MultinomialNB:
        """Naive Bayes classifier for multinomial models."""

        def __init__(self, alpha: float = 1.0):
            self.alpha = alpha

        def fit_predict(self, X_train: Tensor, y_train: Tensor, X_test: Tensor) -> Tensor:
            return _mnb(X_train, y_train, X_test, smoothing=self.alpha)

    class BernoulliNB:
        """Naive Bayes classifier for multivariate Bernoulli models."""

        def __init__(self, alpha: float = 1.0, binarize: float | None = 0.0):
            self.alpha = alpha
            self.binarize = binarize

        def fit_predict(self, X_train: Tensor, y_train: Tensor, X_test: Tensor) -> Tensor:
            return _bnb(X_train, y_train, X_test, smoothing=self.alpha, binarize_threshold=self.binarize)


# ============================================================================
# decomposition
# ============================================================================


class _Decomposition:
    """Namespace mimicking sklearn.decomposition."""

    class NMF:
        """Non-Negative Matrix Factorization."""

        def __init__(self, n_components: int = 2, max_iter: int = 200, tol: float = 1e-4):
            self.n_components = n_components
            self.max_iter = max_iter
            self.tol = tol
            self.components_ = None

        def fit_transform(self, X: Tensor) -> Tensor:
            W, H = _nmf(
                X,
                component_count=self.n_components,
                maximum_iterations=self.max_iter,
                convergence_tolerance=self.tol,
            )
            self.components_ = H
            return W


# ============================================================================
# feature_extraction.text
# ============================================================================


class _FeatureExtractionText:
    """Namespace mimicking sklearn.feature_extraction.text."""

    class CountVectorizer:
        """Convert a collection of text documents to a matrix of token counts."""

        def fit_transform(self, raw_documents: list[str]) -> Tensor:
            return _count_vectorizer(raw_documents)

    class TfidfVectorizer:
        """Convert a collection of raw documents to a matrix of TF-IDF features."""

        def fit_transform(self, raw_documents: list[str]) -> Tensor:
            return _tfidf_vectorizer(raw_documents)


class _FeatureExtraction:
    """Namespace mimicking sklearn.feature_extraction."""
    text = _FeatureExtractionText()


# ============================================================================
# Public namespace assembly
# ============================================================================

preprocessing = _Preprocessing()
metrics = _Metrics()
neighbors = _Neighbors()
naive_bayes = _NaiveBayes()
decomposition = _Decomposition()
feature_extraction = _FeatureExtraction()