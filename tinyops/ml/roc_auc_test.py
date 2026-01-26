import numpy as np
import pytest
from sklearn.metrics import roc_auc_score as sklearn_roc_auc_score
from tinygrad import Tensor

from tinyops._core import assert_close
from tinyops.ml.roc_auc import roc_auc


def test_roc_auc_score_basic():
    y_true_np = np.array([0, 0, 1, 1])
    y_scores_np = np.array([0.1, 0.4, 0.35, 0.8])

    y_true_tiny = Tensor(y_true_np)
    y_scores_tiny = Tensor(y_scores_np)

    auc_tiny = roc_auc(y_true_tiny, y_scores_tiny)
    auc_sklearn = sklearn_roc_auc_score(y_true_np, y_scores_np)

    assert_close(auc_tiny, Tensor([auc_sklearn]))


def test_roc_auc_score_multiclass():
    y_true_np = np.array([0, 0, 1, 1, 2, 2])
    y_scores_np = np.array(
        [[0.9, 0.05, 0.05], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8], [0.1, 0.2, 0.7]]
    )

    with pytest.raises(ValueError):
        y_true_tiny = Tensor(y_true_np)
        y_scores_tiny = Tensor(y_scores_np)
        roc_auc(y_true_tiny, y_scores_tiny)


def test_roc_auc_score_single_class():
    y_true_np = np.array([0, 0, 0, 0])
    y_scores_np = np.array([0.1, 0.4, 0.35, 0.8])

    with pytest.raises(ValueError):
        y_true_tiny = Tensor(y_true_np)
        y_scores_tiny = Tensor(y_scores_np)
        roc_auc(y_true_tiny, y_scores_tiny)
