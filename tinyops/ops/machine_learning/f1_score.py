from tinygrad import Tensor

from tinyops.ops.machine_learning._metrics import _safe_divide
from tinyops.ops.machine_learning.precision_score import precision_score
from tinyops.ops.machine_learning.recall_score import recall_score


def f1_score(true_labels: Tensor, predicted_labels: Tensor) -> Tensor:
    """Compute the F1 score (harmonic mean of precision and recall).

    Args:
        true_labels: Ground truth binary labels.
        predicted_labels: Predicted binary labels.

    Returns:
        F1 score.
    """
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    denominator = precision + recall
    return 2 * _safe_divide(precision * recall, denominator)
