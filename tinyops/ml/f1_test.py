import numpy as np
from sklearn.metrics import f1_score

from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.ml.f1 import f1

def test_f1():
  y_true_np = np.array([0, 1, 1, 0])
  y_pred_np = np.array([0, 1, 0, 0])

  y_true = Tensor(y_true_np)
  y_pred = Tensor(y_pred_np)

  tinyops_f1 = f1(y_true, y_pred)
  sklearn_f1 = f1_score(y_true_np, y_pred_np)

  assert_close(tinyops_f1, Tensor(sklearn_f1))
