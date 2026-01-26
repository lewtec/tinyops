import numpy as np
from sklearn.metrics import recall_score
from tinygrad import Tensor

from tinyops._core import assert_close
from tinyops.ml.recall import recall


def test_recall():
    y_true_np = np.array([0, 1, 1, 0])
    y_pred_np = np.array([0, 1, 0, 0])

    y_true = Tensor(y_true_np)
    y_pred = Tensor(y_pred_np)

    tinyops_recall = recall(y_true, y_pred)
    sklearn_recall = recall_score(y_true_np, y_pred_np)

    assert_close(tinyops_recall, Tensor(sklearn_recall))
