from tinygrad import Tensor, dtypes

from tinyops.ops.machine_learning._tree import _traverse_tree


def decision_tree_classifier(samples: Tensor, tree: dict) -> Tensor:
    """Predict class labels using a pre-trained decision tree.

    Args:
        samples: Input samples (n_samples, n_features).
        tree: Dictionary containing tree structure tensors:
            'children_left', 'children_right', 'feature', 'threshold',
            'value', 'max_depth', 'n_classes'.

    Returns:
        Predicted class labels (n_samples,).
    """
    node_predictions = tree["value"].argmax(axis=1).cast(dtypes.int32)
    node_indices = _traverse_tree(samples, tree)
    return node_predictions.gather(0, node_indices)
