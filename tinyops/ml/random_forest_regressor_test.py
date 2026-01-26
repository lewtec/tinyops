import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from tinygrad import Tensor, dtypes

from tinyops._core import assert_close
from tinyops.ml.random_forest_regressor import random_forest_regressor


# This helper function is defined in the decision tree test, but we redefine it here
# to make this test file self-contained.
def _convert_tree_to_tensor_dict(tree_model):
    """Converts a scikit-learn tree object to a dictionary of tinygrad Tensors."""
    tree = tree_model.tree_
    return {
        "children_left": Tensor(tree.children_left.astype(np.int32), dtype=dtypes.int32),
        "children_right": Tensor(tree.children_right.astype(np.int32), dtype=dtypes.int32),
        "feature": Tensor(tree.feature.astype(np.int32), dtype=dtypes.int32),
        "threshold": Tensor(tree.threshold.astype(np.float32)),
        "value": Tensor(tree.value.squeeze().astype(np.float32)),
        "max_depth": tree.max_depth,
    }


@pytest.mark.parametrize(
    "n_samples, n_features, n_estimators, max_depth, random_state",
    [
        (100, 10, 10, 5, 42),
        (50, 5, 5, 3, 123),
        (200, 15, 20, 10, 0),
    ],
)
def test_random_forest_regressor_parity(n_samples, n_features, n_estimators, max_depth, random_state):
    # 1. Generate data and train a scikit-learn model
    X_np, y_np = make_regression(
        n_samples=n_samples, n_features=n_features, n_informative=n_features // 2, random_state=random_state
    )
    X_np = X_np.astype(np.float32)
    y_np = y_np.astype(np.float32)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        bootstrap=False,  # Disable bootstrap for deterministic comparison
    )
    model.fit(X_np, y_np)

    # 2. Get the expected predictions from scikit-learn
    expected_predictions = model.predict(X_np)

    # 3. Convert the forest (list of trees) and input data for tinyops
    trees_list = [_convert_tree_to_tensor_dict(estimator) for estimator in model.estimators_]
    X_tiny = Tensor(X_np)

    # 4. Get the actual predictions from tinyops
    actual_predictions = random_forest_regressor(X_tiny, trees_list)

    # 5. Assert that the predictions are identical
    assert_close(actual_predictions, expected_predictions, atol=1e-6, rtol=1e-6)
