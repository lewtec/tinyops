from tinygrad import Tensor, dtypes


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
    node_indices = Tensor.zeros(samples.shape[0], dtype=dtypes.int32)
    node_predictions = tree["value"].argmax(axis=1).cast(dtypes.int32)

    for _ in range(tree["max_depth"]):
        features = tree["feature"].gather(0, node_indices)
        thresholds = tree["threshold"].gather(0, node_indices)
        is_leaf = features < 0

        feature_indices = Tensor.where(is_leaf, 0, features).cast(dtypes.int32).unsqueeze(1)
        sample_values = samples.gather(1, feature_indices).squeeze(1)

        go_left = sample_values <= thresholds
        left_children = tree["children_left"].gather(0, node_indices)
        right_children = tree["children_right"].gather(0, node_indices)
        next_nodes = Tensor.where(go_left, left_children, right_children)
        node_indices = Tensor.where(is_leaf, node_indices, next_nodes).cast(dtypes.int32)

    return node_predictions.gather(0, node_indices)
