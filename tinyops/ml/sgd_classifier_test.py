import numpy as np
from sklearn.linear_model import SGDClassifier
from tinygrad import Tensor

from tinyops._core import assert_close, assert_one_kernel
from tinyops.ml.sgd_classifier import sgd_classifier


def test_sgd_classifier_step_hinge():
    n_features = 10
    # Setup: Create and realize inputs for a single sample
    X_np = np.random.randn(n_features).astype(np.float32)
    y_np = np.random.choice([-1.0, 1.0]).astype(np.float32)
    y_sklearn = np.array([y_np])

    weights_np = np.random.randn(n_features).astype(np.float32)
    bias_np = np.random.randn(1).astype(np.float32)[0]

    X = Tensor(X_np).realize()
    y = Tensor([y_np]).realize()
    weights = Tensor(weights_np).realize()
    bias = Tensor([bias_np]).realize()

    # Scikit-learn reference for a single sample update
    clf = SGDClassifier(
        loss="hinge", penalty=None, learning_rate="constant", eta0=0.01, max_iter=1, tol=None, shuffle=False
    )
    clf.coef_ = weights_np.reshape(1, -1)
    clf.intercept_ = np.array([bias_np])
    clf.classes_ = np.array([-1.0, 1.0])
    clf.partial_fit(X_np.reshape(1, -1), y_sklearn)

    expected_weights = clf.coef_.flatten()
    expected_bias = clf.intercept_[0]

    @assert_one_kernel
    def run_kernel():
        new_weights, new_bias = sgd_classifier(X, y, weights, bias, lr=0.01)
        return new_weights.cat(new_bias).realize()

    result_combined = run_kernel()
    result_weights = result_combined[:-1]
    result_bias = result_combined[-1]

    # Validation of value
    assert_close(result_weights, expected_weights)
    assert_close(result_bias, expected_bias)
