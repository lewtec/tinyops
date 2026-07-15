from tinygrad import Tensor


def _true_positives(true_labels: Tensor, predicted_labels: Tensor) -> Tensor:
    """Calculate true positives for binary classification."""
    return (true_labels * predicted_labels).sum()
