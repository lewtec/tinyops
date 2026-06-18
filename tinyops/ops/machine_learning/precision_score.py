from tinygrad import Tensor


def precision_score(true_labels: Tensor, predicted_labels: Tensor) -> Tensor:
    """Compute binary classification precision.

    Args:
        true_labels: Ground truth binary labels.
        predicted_labels: Predicted binary labels.

    Returns:
        Precision score (true positives / predicted positives).
    """
    true_positives = (true_labels * predicted_labels).sum()
    predicted_positives = predicted_labels.sum()
    return true_positives / Tensor.where(predicted_positives == 0, 1, predicted_positives)
