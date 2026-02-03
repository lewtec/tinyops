import numpy as np
from tinygrad import Tensor, dtypes


def confusion_matrix(y_true: Tensor, y_pred: Tensor, labels: list[int] | None = None) -> Tensor:
    """
    Compute confusion matrix to evaluate the accuracy of a classification.

    By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
    is equal to the number of observations known to be in group :math:`i` and
    predicted to be in group :math:`j`.

    This implementation avoids explicit loops over samples by using a matrix multiplication
    trick with boolean masks.

    Args:
        y_true: Ground truth (correct) target values. Shape (n_samples,).
        y_pred: Estimated targets as returned by a classifier. Shape (n_samples,).
        labels: List of labels to index the matrix. This may be used to reorder or
            select a subset of labels. If None, labels are determined from the unique
            values in `y_true` and `y_pred` sorted in ascending order.

    Returns:
        Confusion matrix of shape (n_classes, n_classes). The rows represent the
        true classes, and the columns represent the predicted classes.
    """
    if labels is None:
        y_true_np = y_true.numpy()
        y_pred_np = y_pred.numpy()
        labels_list = sorted(list(np.unique(np.concatenate((y_true_np, y_pred_np)))))
    else:
        labels_list = labels

    labels_tensor = Tensor(labels_list, requires_grad=False)

    # Create boolean matrices indicating where y_true/y_pred match each label.
    # Shape: (n_samples, n_labels)
    y_true_eq_labels = y_true.unsqueeze(1) == labels_tensor.unsqueeze(0)
    y_pred_eq_labels = y_pred.unsqueeze(1) == labels_tensor.unsqueeze(0)

    # The confusion matrix is the matrix product of the transposed true labels matrix
    # and the predicted labels matrix. This efficiently counts all pairs.
    # (n_labels, n_samples) @ (n_samples, n_labels) -> (n_labels, n_labels)
    cm = y_true_eq_labels.transpose(0, 1).cast(dtypes.int64) @ y_pred_eq_labels.cast(dtypes.int64)

    return cm
