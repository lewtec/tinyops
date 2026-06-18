from tinygrad import Tensor


def accuracy_score(true_labels: Tensor, predicted_labels: Tensor) -> Tensor:
    """Compute classification accuracy.

    Args:
        true_labels: Ground truth labels.
        predicted_labels: Predicted labels.

    Returns:
        Fraction of correctly classified samples.
    """
    return (true_labels == predicted_labels).mean()
