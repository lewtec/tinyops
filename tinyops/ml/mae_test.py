import numpy as np
from sklearn.metrics import mean_absolute_error
from tinygrad import Tensor

from tinyops._core import assert_close
from tinyops.ml.mae import mae


def test_mae():
    y_true_np = np.array([3, -0.5, 2, 7])
    y_pred_np = np.array([2.5, 0.0, 2, 8])

    y_true_tiny = Tensor(y_true_np)
    y_pred_tiny = Tensor(y_pred_np)

    mae_tiny = mae(y_true_tiny, y_pred_tiny)
    mae_sklearn = mean_absolute_error(y_true_np, y_pred_np)
    assert_close(mae_tiny, Tensor([mae_sklearn]))
