from tinygrad import Tensor
from tinyops.ml._tree_utils import _traverse_tree

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
    # Traverse the tree to find the leaf node index for each sample.
    node_indices = _traverse_tree(X, tree)

    # After traversal, node_indices contains the leaf node ID for each sample.
    # Gather the final prediction values from the leaf nodes.
    final_values = tree['value'].gather(0, node_indices)

    return final_values
