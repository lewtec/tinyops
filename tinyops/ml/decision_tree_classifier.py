from tinygrad import Tensor, dtypes

def decision_tree_classifier(X: Tensor, tree: dict) -> Tensor:
    """
    Predicts class labels for samples in X using a pre-trained decision tree.
    The tree structure is expected to be compatible with scikit-learn's internal tree representation.

    Args:
        X: Input samples of shape (n_samples, n_features).
        tree: A dictionary containing the tree structure as Tensors:
              'children_left', 'children_right', 'feature', 'threshold', 'value', 'max_depth', 'n_classes'.

    Returns:
        A Tensor of shape (n_samples,) with the predicted class labels.
    """
    node_indices = Tensor.zeros(X.shape[0], dtype=dtypes.int32)

    # Pre-calculate the predicted class for every node. This simplifies the final gather.
    node_predictions = tree['value'].argmax(axis=1).cast(dtypes.int32)

    for _ in range(tree['max_depth']):
        features = tree['feature'].gather(0, node_indices)
        thresholds = tree['threshold'].gather(0, node_indices)
        is_leaf = features < 0

        feature_indices_for_gather = Tensor.where(is_leaf, 0, features).cast(dtypes.int32).unsqueeze(1)
        sample_feature_values = X.gather(1, feature_indices_for_gather).squeeze(1)

        go_left = sample_feature_values <= thresholds
        children_left = tree['children_left'].gather(0, node_indices)
        children_right = tree['children_right'].gather(0, node_indices)
        next_nodes = Tensor.where(go_left, children_left, children_right)

        node_indices = Tensor.where(is_leaf, node_indices, next_nodes).cast(dtypes.int32)

    # Gather the final predictions from the pre-calculated leaf node predictions.
    return node_predictions.gather(0, node_indices)
