import numpy as np
import pytest
from sklearn.preprocessing import PolynomialFeatures as SklearnPolynomialFeatures
from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.ml.polynomial_features import polynomial_features as tinyops_polynomial_features

@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("interaction_only", [True, False])
@pytest.mark.parametrize("include_bias", [True, False])
def test_polynomial_features(degree, interaction_only, include_bias):
    X_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
    X_tiny = Tensor(X_np)

    # Sklearn's PolynomialFeatures
    sklearn_poly = SklearnPolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
    sklearn_result = sklearn_poly.fit_transform(X_np)

    tinyops_result = tinyops_polynomial_features(X_tiny, degree=degree, interaction_only=interaction_only, include_bias=include_bias)

    assert_close(tinyops_result, sklearn_result)

def test_polynomial_features_degree_0():
    X_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
    X_tiny = Tensor(X_np)

    sklearn_poly = SklearnPolynomialFeatures(degree=0)
    sklearn_result = sklearn_poly.fit_transform(X_np)

    tinyops_result = tinyops_polynomial_features(X_tiny, degree=0)

    assert_close(tinyops_result, sklearn_result)

def test_polynomial_features_degree_1():
    X_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
    X_tiny = Tensor(X_np)

    sklearn_poly = SklearnPolynomialFeatures(degree=1)
    sklearn_result = sklearn_poly.fit_transform(X_np)

    tinyops_result = tinyops_polynomial_features(X_tiny, degree=1)

    assert_close(tinyops_result, sklearn_result)

def test_polynomial_features_exceeds_limit():
    # n_features = 30, degree = 6 generates > 1M features
    # math.comb(30 + 6 - 1, 6) = 1,623,160
    n_samples = 1
    n_features = 30
    X = Tensor.zeros(n_samples, n_features)

    with pytest.raises(ValueError, match="Output features .* exceeds limit"):
        tinyops_polynomial_features(X, degree=6, interaction_only=False)
