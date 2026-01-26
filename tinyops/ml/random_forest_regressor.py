from tinygrad import Tensor

from tinyops.ml.decision_tree_regressor import decision_tree_regressor


def random_forest_regressor(X: Tensor, trees: list[dict]) -> Tensor:
    """
    Predicts regression target for samples in X using a pre-trained random forest.

    Args:
        X: Input samples of shape (n_samples, n_features).
        trees: A list of dictionaries, where each dictionary represents a trained decision tree.

    Returns:
        A Tensor of shape (n_samples,) with the predicted regression values.
    """
    # Get predictions from each tree in the forest.
    all_predictions = [decision_tree_regressor(X, tree) for tree in trees]

    # Stack all predictions into a single tensor of shape (n_samples, n_trees).
    stacked_predictions = Tensor.stack(all_predictions, dim=1)

    # Return the mean of the predictions across all trees.
    return stacked_predictions.mean(axis=1)
