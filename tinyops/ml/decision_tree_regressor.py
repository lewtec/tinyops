from tinygrad import Tensor, dtypes

def decision_tree_regressor(X: Tensor, tree: dict) -> Tensor:
    """
    Predicts regression target for samples in X using a pre-trained decision tree.
    The tree structure is expected to be compatible with scikit-learn's internal tree representation.

    Args:
        X: Input samples of shape (n_samples, n_features).
        tree: A dictionary containing the tree structure as Tensors:
              'children_left', 'children_right', 'feature', 'threshold', 'value', 'max_depth'.

    Returns:
        A Tensor of shape (n_samples,) with the predicted regression values.
    """
    node_indices = Tensor.zeros(X.shape[0], dtype=dtypes.int32)

    if tree['max_depth'] > 0:
      for _ in range(tree['max_depth']):
          # Get properties of the current nodes for each sample using gather
          features = tree['feature'].gather(0, node_indices)
          thresholds = tree['threshold'].gather(0, node_indices)

          # Determine if the current nodes are leaves. In sklearn, feature is < 0 for leaves.
          is_leaf = features < 0

          # Get the feature values from X for each sample to perform the split.
          # Use a dummy index (0) for leaves to avoid invalid gathers, as their feature index is negative.
          feature_indices_for_gather = Tensor.where(is_leaf, 0, features).cast(dtypes.int32).unsqueeze(1)
          sample_feature_values = X.gather(1, feature_indices_for_gather).squeeze(1)

          # Decide whether to go left or right down the tree
          go_left = sample_feature_values <= thresholds

          # Get the IDs of the potential next nodes
          children_left = tree['children_left'].gather(0, node_indices)
          children_right = tree['children_right'].gather(0, node_indices)

          # Choose the next node based on the split condition
          next_nodes = Tensor.where(go_left, children_left, children_right)

          # Update node_indices for non-leaf nodes only. Leaves keep their own index.
          # Crucially, cast the result back to int32 for the next iteration's gather.
          node_indices = Tensor.where(is_leaf, node_indices, next_nodes).cast(dtypes.int32)

    # After traversal, node_indices contains the leaf node ID for each sample.
    # Gather the final prediction values from the leaf nodes.
    final_values = tree['value'].gather(0, node_indices)

    return final_values
