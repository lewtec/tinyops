import numpy as np
import pytest
from tinygrad import Tensor
from sklearn.linear_model import Ridge as SklearnRidge
from tinyops._core import assert_close
from tinyops.ml.ridge import ridge

@pytest.mark.parametrize("n_samples, n_features, alpha", [(100, 5, 1.0), (50, 10, 0.5), (200, 3, 10.0)])
def test_ridge_regression(n_samples, n_features, alpha):
    # Setup: Create and realize inputs
    X_np = np.random.randn(n_samples, n_features).astype(np.float32)
    true_weights = np.random.randn(n_features).astype(np.float32)
    true_bias = np.float32(2.5)
    y_np = X_np @ true_weights + true_bias + np.random.normal(0, 0.1, n_samples).astype(np.float32)

    X_tg = Tensor(X_np)
    y_tg = Tensor(y_np)

    # tinyops implementation
    weights_tg = ridge(X_tg, y_tg, alpha=alpha)
    weights_np = weights_tg.numpy()

    # scikit-learn implementation
    model = SklearnRidge(alpha=alpha, fit_intercept=True)
    model.fit(X_np, y_np)

    # Compare results
    assert_close(weights_np[:-1], model.coef_, atol=1e-5, rtol=1e-5)
    assert_close(weights_np[-1], model.intercept_, atol=1e-5, rtol=1e-5)
