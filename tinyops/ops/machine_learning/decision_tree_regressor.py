from tinygrad import Tensor

from tinyops.ops.machine_learning._tree import _traverse_tree


def decision_tree_regressor(samples: Tensor, tree: dict) -> Tensor:
    """Predict regression targets using a pre-trained decision tree.

    Args:
        samples: Input samples (n_samples, n_features).
        tree: Dictionary containing tree structure tensors.

    Returns:
        Predicted values (n_samples,).
    """
    node_indices = _traverse_tree(samples, tree)
    return tree["value"].gather(0, node_indices)
