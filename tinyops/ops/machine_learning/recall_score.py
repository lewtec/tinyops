from tinygrad import Tensor

from tinyops.ops.machine_learning._metrics import _true_positives


def recall_score(true_labels: Tensor, predicted_labels: Tensor) -> Tensor:
    """Compute binary classification recall.

    Args:
        true_labels: Ground truth binary labels.
        predicted_labels: Predicted binary labels.

    Returns:
        Recall score (true positives / actual positives).
    """
    true_positives = _true_positives(true_labels, predicted_labels)
    actual_positives = true_labels.sum()
    return true_positives / Tensor.where(actual_positives == 0, 1, actual_positives)
