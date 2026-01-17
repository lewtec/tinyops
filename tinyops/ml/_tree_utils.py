from tinygrad import Tensor, dtypes

def _traverse_tree(X: Tensor, tree: dict) -> Tensor:
    """
    Traverses a decision tree and returns the leaf node indices for each sample in X.
    """
    node_indices = Tensor.zeros(X.shape[0], dtype=dtypes.int32)

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

    return node_indices
