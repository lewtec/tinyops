import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from tinygrad import Tensor, dtypes

from tinyops._core import assert_close
from tinyops.ml.decision_tree_regressor import decision_tree_regressor


def _convert_tree_to_tensor_dict(tree_model):
    """Converts a scikit-learn tree object to a dictionary of tinygrad Tensors."""
    tree = tree_model.tree_
    return {
        "children_left": Tensor(tree.children_left.astype(np.int32), dtype=dtypes.int32),
        "children_right": Tensor(tree.children_right.astype(np.int32), dtype=dtypes.int32),
        "feature": Tensor(tree.feature.astype(np.int32), dtype=dtypes.int32),
        "threshold": Tensor(tree.threshold.astype(np.float32)),
        # Squeeze completely to get a 1D tensor of values for simplicity.
        "value": Tensor(tree.value.squeeze().astype(np.float32)),
        "max_depth": tree.max_depth,
    }


@pytest.mark.parametrize(
    "n_samples, n_features, max_depth, random_state",
    [
        (100, 10, 5, 42),
        (50, 5, 3, 123),
        (200, 2, 10, 0),
        (10, 2, 2, 99),
    ],
)
def test_decision_tree_regressor_parity(n_samples, n_features, max_depth, random_state):
    # 1. Generate data and train a scikit-learn model
    X_np, y_np = make_regression(
        n_samples=n_samples, n_features=n_features, n_informative=n_features // 2, random_state=random_state
    )
    X_np = X_np.astype(np.float32)
    y_np = y_np.astype(np.float32)

    model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    model.fit(X_np, y_np)

    # 2. Get the expected predictions from scikit-learn
    expected_predictions = model.predict(X_np)

    # 3. Convert the tree and input data for tinyops
    tree_dict = _convert_tree_to_tensor_dict(model)
    X_tiny = Tensor(X_np)

    # 4. Get the actual predictions from tinyops
    actual_predictions = decision_tree_regressor(X_tiny, tree_dict)

    # 5. Assert that the predictions are identical
    assert_close(actual_predictions, expected_predictions, atol=1e-6, rtol=1e-6)
