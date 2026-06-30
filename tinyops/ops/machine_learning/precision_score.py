from tinygrad import Tensor

from tinyops.ops.machine_learning._classification_utils import calculate_binary_components


def precision_score(true_labels: Tensor, predicted_labels: Tensor) -> Tensor:
    """Compute binary classification precision.

    Args:
        true_labels: Ground truth binary labels.
        predicted_labels: Predicted binary labels.

    Returns:
        Precision score (true positives / predicted positives).
    """
    true_positives, predicted_positives, _ = calculate_binary_components(true_labels, predicted_labels)
    return true_positives / Tensor.where(predicted_positives == 0, 1, predicted_positives)
