from tinygrad import Tensor

from tinyops.ml.decision_tree_classifier import decision_tree_classifier


def random_forest_classifier(X: Tensor, trees: list[dict]) -> Tensor:
    """
    Predicts class labels for samples in X using a pre-trained random forest.

    Args:
        X: Input samples of shape (n_samples, n_features).
        trees: A list of dictionaries, where each dictionary represents a trained decision tree.

    Returns:
        A Tensor of shape (n_samples,) with the predicted class labels.
    """
    # Get predictions from each tree in the forest.
    all_predictions = [decision_tree_classifier(X, tree) for tree in trees]

    # Stack all predictions into a single tensor of shape (n_samples, n_trees).
    stacked_predictions = Tensor.stack(all_predictions, dim=1)

    n_classes = trees[0]["n_classes"]

    # Manually calculate vote counts for each class to avoid using one_hot.
    # This is a more robust way to compute the mode.
    vote_counts_list = []
    for i in range(n_classes):
        # Create a boolean mask where predictions match the current class `i`.
        is_class_i = stacked_predictions == i
        # Sum the boolean mask (True=1, False=0) across the trees to get the vote count.
        counts_for_class_i = is_class_i.sum(axis=1).unsqueeze(1)
        vote_counts_list.append(counts_for_class_i)

    # Concatenate the vote counts for all classes into a single tensor.
    vote_counts = Tensor.cat(*vote_counts_list, dim=1)

    # The final prediction is the class with the most votes.
    return vote_counts.argmax(axis=1)
