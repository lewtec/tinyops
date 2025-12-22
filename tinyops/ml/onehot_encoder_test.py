import numpy as np
import pytest
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.ml.onehot_encoder import onehot_encoder as tinyops_onehot_encoder

def test_onehot_encoder_2d():
    X_np = np.array([[0, 1, 2], [2, 0, 1]], dtype=np.int32)
    X_tiny = Tensor(X_np)

    # Sklearn's OneHotEncoder
    sklearn_encoder = SklearnOneHotEncoder(categories='auto', sparse_output=False, dtype=np.float32)
    sklearn_result = sklearn_encoder.fit_transform(X_np)

    tinyops_result = tinyops_onehot_encoder(X_tiny)

    assert_close(tinyops_result, sklearn_result)

def test_onehot_encoder_1d():
    X_np = np.array([0, 1, 2, 1, 0], dtype=np.int32)
    X_tiny = Tensor(X_np)

    # Sklearn's OneHotEncoder expects 2D input
    X_np_2d = X_np.reshape(-1, 1)

    sklearn_encoder = SklearnOneHotEncoder(categories='auto', sparse_output=False, dtype=np.float32)
    sklearn_result = sklearn_encoder.fit_transform(X_np_2d)

    tinyops_result = tinyops_onehot_encoder(X_tiny)

    assert_close(tinyops_result, sklearn_result)
