import numpy as np
import pytest
from sklearn.preprocessing import Binarizer
from tinygrad import Tensor

from tinyops._core import assert_close
from tinyops.ml.binarizer import binarizer as tinyops_binarizer


@pytest.mark.parametrize("threshold", [0.0, 0.5, -0.5])
def test_binarizer(threshold):
    X_np = np.array([[-2.0, 1.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float32)
    X_tiny = Tensor(X_np)

    # Sklearn's Binarizer
    sklearn_binarizer = Binarizer(threshold=threshold)
    sklearn_result = sklearn_binarizer.transform(X_np)

    tinyops_result = tinyops_binarizer(X_tiny, threshold=threshold)

    assert_close(tinyops_result, sklearn_result)
