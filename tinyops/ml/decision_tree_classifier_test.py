import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from tinygrad import Tensor, dtypes
from tinyops.ml.decision_tree_classifier import decision_tree_classifier
from tinyops._core import assert_close

def _convert_tree_to_tensor_dict(tree_model):
    """Converts a scikit-learn tree object to a dictionary of tinygrad Tensors."""
    tree = tree_model.tree_
    return {
        'children_left': Tensor(tree.children_left.astype(np.int32), dtype=dtypes.int32),
        'children_right': Tensor(tree.children_right.astype(np.int32), dtype=dtypes.int32),
        'feature': Tensor(tree.feature.astype(np.int32), dtype=dtypes.int32),
        'threshold': Tensor(tree.threshold.astype(np.float32)),
        'value': Tensor(tree.value.squeeze(1).astype(np.float32)),
        'max_depth': tree.max_depth,
        'n_classes': tree.n_classes[0],
    }

@pytest.mark.parametrize("n_samples, n_features, n_classes, max_depth, random_state", [
    (100, 10, 3, 5, 42),
    (50, 5, 2, 3, 123),
    (200, 15, 5, 10, 0),
    (20, 4, 2, 2, 99),
])
def test_decision_tree_classifier_parity(n_samples, n_features, n_classes, max_depth, random_state):
    # 1. Generate data and train a scikit-learn model
    X_np, y_np = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random_state
    )
    X_np = X_np.astype(np.float32)

    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(X_np, y_np)

    # 2. Get the expected predictions and cast to int32 to match tinygrad's argmax output dtype
    expected_predictions = model.predict(X_np).astype(np.int32)

    # 3. Convert the tree and input data for tinyops
    tree_dict = _convert_tree_to_tensor_dict(model)
    X_tiny = Tensor(X_np)

    # 4. Get the actual predictions from tinyops
    actual_predictions = decision_tree_classifier(X_tiny, tree_dict)

    # 5. Assert that the predictions are identical
    assert_close(actual_predictions, expected_predictions)
