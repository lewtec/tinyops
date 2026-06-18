import numpy as np
from tinygrad import Tensor, dtypes


def confusion_matrix(
    true_labels: Tensor,
    predicted_labels: Tensor,
    label_values: list[int] | None = None,
) -> Tensor:
    """Compute the confusion matrix for classification evaluation.

    Args:
        true_labels: Ground truth labels.
        predicted_labels: Predicted labels.
        label_values: Explicit list of label values. If None, inferred from data.

    Returns:
        Confusion matrix of shape (n_classes, n_classes).
    """
    if label_values is None:
        true_np = true_labels.numpy()
        pred_np = predicted_labels.numpy()
        label_values = sorted(list(np.unique(np.concatenate((true_np, pred_np)))))

    labels_tensor = Tensor(label_values, requires_grad=False)

    true_matches = true_labels.unsqueeze(1) == labels_tensor.unsqueeze(0)
    pred_matches = predicted_labels.unsqueeze(1) == labels_tensor.unsqueeze(0)

    return true_matches.transpose(0, 1).cast(dtypes.int64) @ pred_matches.cast(dtypes.int64)
