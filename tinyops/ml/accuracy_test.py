import numpy as np
from sklearn.metrics import accuracy_score

from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.ml.accuracy import accuracy

def test_accuracy():
  y_true_np = np.array([0, 1, 1, 0])
  y_pred_np = np.array([0, 1, 0, 0])

  y_true = Tensor(y_true_np)
  y_pred = Tensor(y_pred_np)

  tinyops_accuracy = accuracy(y_true, y_pred)
  sklearn_accuracy = accuracy_score(y_true_np, y_pred_np)

  assert_close(tinyops_accuracy, Tensor(sklearn_accuracy))
