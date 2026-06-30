from tinygrad import Tensor

from tinyops.ops.machine_learning._classification_utils import calculate_binary_components


def f1_score(true_labels: Tensor, predicted_labels: Tensor) -> Tensor:
    """Compute the F1 score (harmonic mean of precision and recall).

    Args:
        true_labels: Ground truth binary labels.
        predicted_labels: Predicted binary labels.

    Returns:
        F1 score.
    """
    true_positives, predicted_positives, actual_positives = calculate_binary_components(true_labels, predicted_labels)

    precision = true_positives / Tensor.where(predicted_positives == 0, 1, predicted_positives)
    recall = true_positives / Tensor.where(actual_positives == 0, 1, actual_positives)

    denominator = precision + recall
    return 2 * (precision * recall) / Tensor.where(denominator == 0, 1, denominator)
