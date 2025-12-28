import numpy as np
import pytest
from tinygrad import Tensor
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from tinyops._core import assert_close
from tinyops.ml.linear_regression import linear_regression

@pytest.mark.parametrize("n_samples, n_features", [(100, 5)])
def test_linear_regression(n_samples, n_features):
    # Setup: Create and realize inputs
    X_np = np.random.randn(n_samples, n_features).astype(np.float32)
    # Generate a linear relationship for y
    true_weights = np.random.randn(n_features).astype(np.float32)
    true_bias = np.float32(2.5)
    y_np = X_np @ true_weights + true_bias + np.random.normal(0, 0.1, n_samples).astype(np.float32)

    X_tg = Tensor(X_np)
    y_tg = Tensor(y_np)

    # tinyops implementation
    weights_tg = linear_regression(X_tg, y_tg)
    weights_np = weights_tg.numpy()

    # scikit-learn implementation
    model = SklearnLinearRegression()
    model.fit(X_np, y_np)

    # Compare results
    # The last element of our weights is the bias/intercept
    assert_close(weights_np[:-1], model.coef_, atol=1e-5, rtol=1e-5)
    assert_close(weights_np[-1], model.intercept_, atol=1e-5, rtol=1e-5)
