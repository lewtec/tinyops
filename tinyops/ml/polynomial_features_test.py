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

def test_polynomial_features_invalid_degree():
    X_tiny = Tensor.zeros(2, 2)
    with pytest.raises(ValueError, match="Degree must be non-negative"):
        tinyops_polynomial_features(X_tiny, degree=-1)

def test_polynomial_features_dos_prevention():
    # n_features=10, degree=15 => C(24, 15) = 1,307,504 combinations
    # This exceeds the limit of 100,000
    n_features = 10
    X_tiny = Tensor.zeros(1, n_features)

    with pytest.raises(ValueError, match="exceeds the limit"):
        tinyops_polynomial_features(X_tiny, degree=15)
