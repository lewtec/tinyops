from tinygrad import Tensor, dtypes
from tinyops.ml._tree_utils import _traverse_tree

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
    # Pre-calculate the predicted class for every node. This simplifies the final gather.
    node_predictions = tree['value'].argmax(axis=1).cast(dtypes.int32)

    # Traverse the tree to find the leaf node index for each sample.
    node_indices = _traverse_tree(X, tree)

    # Gather the final predictions from the pre-calculated leaf node predictions.
    return node_predictions.gather(0, node_indices)
