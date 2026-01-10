import pytest
import numpy as np
from tinygrad import Tensor
from tinyops.ml.logistic_regression import logistic_regression
from tinyops._core import assert_close
from tinyops._core import assert_one_kernel

@pytest.mark.parametrize("n_samples, n_features", [(100, 10)])
def test_logistic_regression_step(n_samples, n_features):
    # Setup: Create and realize inputs
    X_np = np.random.randn(n_samples, n_features).astype(np.float32)
    y_np = (np.arange(n_samples) % 2).astype(np.float32)

    # Initial parameters
    weights_np = np.random.randn(n_features).astype(np.float32)
    bias_np = np.random.randn(1).astype(np.float32)[0]
    lr = 0.01

    X = Tensor(X_np).realize()
    y = Tensor(y_np).realize()
    weights = Tensor(weights_np).realize()
    bias = Tensor([bias_np]).realize()

    # NumPy reference implementation of one Batch Gradient Descent step
    logits_np = X_np @ weights_np + bias_np
    predictions_np = 1 / (1 + np.exp(-logits_np))
    error_np = predictions_np - y_np
    gradient_weights_np = X_np.T @ error_np / n_samples
    gradient_bias_np = np.mean(error_np)

    expected_weights = weights_np - lr * gradient_weights_np
    expected_bias = bias_np - lr * gradient_bias_np

    @assert_one_kernel
    def run_kernel():
        new_weights, new_bias = logistic_regression(X, y, weights, bias, lr=lr)
        # Combine outputs for a single realization call to satisfy assert_one_kernel
        combined = new_weights.cat(new_bias)
        return combined.realize()

    result_combined = run_kernel()
    result_weights = result_combined[:-1]
    result_bias = result_combined[-1]

    # Validation of value
    assert_close(result_weights, expected_weights)
    assert_close(result_bias, expected_bias)
