import numpy as np
import pytest
from sklearn.metrics import r2_score

from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.ml.r2 import r2

def test_r2_score():
  y_true_np = np.array([3, -0.5, 2, 7])
  y_pred_np = np.array([2.5, 0.0, 2, 8])

  y_true_tiny = Tensor(y_true_np)
  y_pred_tiny = Tensor(y_pred_np)

  # Test with default parameters
  r2_tiny = r2(y_true_tiny, y_pred_tiny)
  r2_sklearn = r2_score(y_true_np, y_pred_np)
  assert_close(r2_tiny, Tensor([r2_sklearn]))

def test_r2_score_constant_y_true():
  y_true_np = np.array([1, 1, 1, 1])
  y_pred_np = np.array([1, 1, 1, 1])

  y_true_tiny = Tensor(y_true_np)
  y_pred_tiny = Tensor(y_pred_np)

  r2_tiny = r2(y_true_tiny, y_pred_tiny)
  r2_sklearn = r2_score(y_true_np, y_pred_np)
  assert_close(r2_tiny, Tensor([r2_sklearn]))
