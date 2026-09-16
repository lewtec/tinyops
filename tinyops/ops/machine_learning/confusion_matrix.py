from tinygrad import Tensor, dtypes

from tinyops.ops._tensor_utils import unique_sorted_values


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
        combined = Tensor.cat(true_labels.reshape(-1), predicted_labels.reshape(-1))
        label_values = unique_sorted_values(combined)

    labels_tensor = Tensor(label_values, requires_grad=False)

    true_matches = true_labels.unsqueeze(1) == labels_tensor.unsqueeze(0)
    pred_matches = predicted_labels.unsqueeze(1) == labels_tensor.unsqueeze(0)

    return true_matches.transpose(0, 1).cast(dtypes.int64) @ pred_matches.cast(dtypes.int64)
