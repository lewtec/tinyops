import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from tinygrad import Tensor, dtypes

from tinyops._core import assert_close
from tinyops.ml.confusion_matrix import confusion_matrix


def test_confusion_matrix_basic():
    y_true_np = np.array([2, 0, 2, 2, 0, 1])
    y_pred_np = np.array([0, 0, 2, 2, 0, 2])

    y_true_tiny = Tensor(y_true_np)
    y_pred_tiny = Tensor(y_pred_np)

    cm_tiny = confusion_matrix(y_true_tiny, y_pred_tiny)
    cm_sklearn = sklearn_confusion_matrix(y_true_np, y_pred_np)

    assert_close(cm_tiny, Tensor(cm_sklearn, dtype=dtypes.int64))


def test_confusion_matrix_with_unseen_labels():
    y_true_np = np.array([0, 1, 2, 3])
    y_pred_np = np.array([0, 1, 2, 4])  # Label 4 is not in labels
    labels = [0, 1, 2]

    y_true_tiny = Tensor(y_true_np)
    y_pred_tiny = Tensor(y_pred_np)

    cm_tiny = confusion_matrix(y_true_tiny, y_pred_tiny, labels=labels)
    cm_sklearn = sklearn_confusion_matrix(y_true_np, y_pred_np, labels=labels)

    assert_close(cm_tiny, Tensor(cm_sklearn, dtype=dtypes.int64))


def test_confusion_matrix_with_labels():
    y_true_np = np.array([2, 0, 2, 2, 0, 1])
    y_pred_np = np.array([0, 0, 2, 2, 0, 2])
    labels = [0, 1, 2]

    y_true_tiny = Tensor(y_true_np)
    y_pred_tiny = Tensor(y_pred_np)

    cm_tiny = confusion_matrix(y_true_tiny, y_pred_tiny, labels=labels)
    cm_sklearn = sklearn_confusion_matrix(y_true_np, y_pred_np, labels=labels)

    assert_close(cm_tiny, Tensor(cm_sklearn, dtype=dtypes.int64))


def test_confusion_matrix_multiclass():
    y_true_np = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred_np = np.array([0, 1, 2, 3, 3, 2, 1, 0])

    y_true_tiny = Tensor(y_true_np)
    y_pred_tiny = Tensor(y_pred_np)

    cm_tiny = confusion_matrix(y_true_tiny, y_pred_tiny)
    cm_sklearn = sklearn_confusion_matrix(y_true_np, y_pred_np)

    assert_close(cm_tiny, Tensor(cm_sklearn, dtype=dtypes.int64))
