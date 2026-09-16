from tinygrad import Tensor

from tinyops.ops.machine_learning.decision_tree_regressor import decision_tree_regressor


def random_forest_regressor(samples: Tensor, trees: list[dict]) -> Tensor:
    """Predict regression targets using a pre-trained random forest.

    Args:
        samples: Input samples (n_samples, n_features).
        trees: List of decision tree dictionaries.

    Returns:
        Mean of tree predictions (n_samples,).
    """
    all_predictions = [decision_tree_regressor(samples, tree) for tree in trees]
    stacked = Tensor.stack(all_predictions, dim=1)
    return stacked.mean(axis=1)
