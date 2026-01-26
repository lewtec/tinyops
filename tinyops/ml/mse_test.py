import numpy as np
from sklearn.metrics import mean_squared_error
from tinygrad import Tensor

from tinyops._core import assert_close
from tinyops.ml.mse import mse


def test_mse():
    y_true_np = np.array([3, -0.5, 2, 7])
    y_pred_np = np.array([2.5, 0.0, 2, 8])

    y_true_tiny = Tensor(y_true_np)
    y_pred_tiny = Tensor(y_pred_np)

    mse_tiny = mse(y_true_tiny, y_pred_tiny)
    mse_sklearn = mean_squared_error(y_true_np, y_pred_np)
    assert_close(mse_tiny, Tensor([mse_sklearn]))
