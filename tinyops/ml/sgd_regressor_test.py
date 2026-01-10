import pytest
import numpy as np
from tinygrad import Tensor
from tinyops.ml.sgd_regressor import sgd_regressor
from tinyops._core import assert_close
from tinyops._core import assert_one_kernel
from sklearn.linear_model import SGDRegressor

def test_sgd_regressor_step():
    n_features = 10
    # Setup: Create and realize inputs for a single sample
    X_np = np.random.randn(n_features).astype(np.float32)
    y_np = np.random.randn(1).astype(np.float32)

    weights_np = np.random.randn(n_features).astype(np.float32)
    bias_np = np.random.randn(1).astype(np.float32)

    X = Tensor(X_np).realize()
    y = Tensor(y_np).realize()
    weights = Tensor(weights_np).realize()
    bias = Tensor(bias_np).realize()

    # Scikit-learn reference for a single sample update
    reg = SGDRegressor(
        loss='squared_error',
        penalty=None,
        learning_rate='constant',
        eta0=0.01,
        max_iter=1,
        tol=None,
        shuffle=False
    )
    reg.coef_ = weights_np
    reg.intercept_ = bias_np
    reg.partial_fit(X_np.reshape(1, -1), y_np)

    expected_weights = reg.coef_.flatten()
    expected_bias = reg.intercept_

    @assert_one_kernel
    def run_kernel():
        new_weights, new_bias = sgd_regressor(X, y, weights, bias, lr=0.01)
        return new_weights.cat(new_bias).realize()

    result_combined = run_kernel()
    result_weights = result_combined[:-1]
    result_bias = result_combined[-1:]

    # Validation of value
    assert_close(result_weights, expected_weights)
    assert_close(result_bias, expected_bias)
