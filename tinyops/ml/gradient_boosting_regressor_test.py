import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from tinygrad import Tensor, dtypes

from tinyops._core import assert_close
from tinyops.ml.decision_tree_regressor_test import _convert_tree_to_tensor_dict
from tinyops.ml.gradient_boosting_regressor import gradient_boosting_regressor


@pytest.mark.parametrize(
    "n_samples, n_features, n_estimators, max_depth, learning_rate, random_state",
    [
        (100, 10, 10, 3, 0.1, 42),
        (200, 5, 50, 5, 0.05, 123),
    ],
)
def test_gradient_boosting_regressor_parity(
    n_samples, n_features, n_estimators, max_depth, learning_rate, random_state
):
    # 1. Generate data and train a scikit-learn model
    X_np, y_np = make_regression(
        n_samples=n_samples, n_features=n_features, n_informative=n_features // 2, random_state=random_state
    )
    X_np = X_np.astype(np.float32)
    y_np = y_np.astype(np.float32)

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    model.fit(X_np, y_np)

    # 2. Get the expected predictions from scikit-learn
    expected_predictions = model.predict(X_np)

    # 3. Convert the model and input data for tinyops
    estimators = [_convert_tree_to_tensor_dict(est[0]) for est in model.estimators_]
    # The default init estimator predicts a constant value (the mean of y_train)
    init_prediction_val = model.init_.predict(X_np)[0]
    init_prediction = Tensor([init_prediction_val], dtype=dtypes.float32)
    X_tiny = Tensor(X_np)

    # 4. Get the actual predictions from tinyops
    actual_predictions = gradient_boosting_regressor(X_tiny, estimators, learning_rate, init_prediction)

    # 5. Assert that the predictions are identical
    assert_close(actual_predictions, expected_predictions, atol=1e-5, rtol=1e-5)
