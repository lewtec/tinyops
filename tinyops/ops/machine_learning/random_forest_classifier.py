from tinygrad import Tensor

from tinyops.ops.machine_learning.decision_tree_classifier import decision_tree_classifier


def random_forest_classifier(samples: Tensor, trees: list[dict]) -> Tensor:
    """Predict class labels using a pre-trained random forest.

    Args:
        samples: Input samples (n_samples, n_features).
        trees: List of decision tree dictionaries.

    Returns:
        Predicted class labels (n_samples,).
    """
    all_predictions = [decision_tree_classifier(samples, tree) for tree in trees]
    stacked = Tensor.stack(all_predictions, dim=1)

    class_count = trees[0]["n_classes"]
    vote_columns = []
    for class_index in range(class_count):
        votes = (stacked == class_index).sum(axis=1).unsqueeze(1)
        vote_columns.append(votes)

    vote_matrix = Tensor.cat(*vote_columns, dim=1)
    return vote_matrix.argmax(axis=1)
