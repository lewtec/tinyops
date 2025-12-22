import numpy as np
import pytest
from sklearn.preprocessing import normalize
from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.ml.normalizer import normalizer as tinyops_normalizer

@pytest.mark.parametrize("norm", ["l1", "l2", "max"])
@pytest.mark.parametrize("axis", [0, 1])
def test_normalizer(norm, axis):
    X_np = np.array([[-2.0, 1.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float32)
    X_tiny = Tensor(X_np)

    # Sklearn's normalize works on rows (axis=1) or columns (axis=0)
    sklearn_result = normalize(X_np, norm=norm, axis=axis)

    tinyops_result = tinyops_normalizer(X_tiny, norm=norm, axis=axis)

    assert_close(tinyops_result, sklearn_result)

def test_normalizer_zero_norm():
    X_np = np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]], dtype=np.float32)
    X_tiny = Tensor(X_np)

    # Sklearn's normalize leaves zero-norm vectors as they are.
    sklearn_result = normalize(X_np, norm='l2', axis=1)

    tinyops_result = tinyops_normalizer(X_tiny, norm='l2', axis=1)

    assert_close(tinyops_result, sklearn_result)
