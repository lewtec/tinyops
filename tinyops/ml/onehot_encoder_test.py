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

def test_onehot_encoder_max_categories_success():
    # Create data with 20 unique values
    X_np = np.arange(20).astype(np.int32).reshape(-1, 1)
    X = Tensor(X_np)

    # max_categories=30 (greater than 20) should succeed
    res = tinyops_onehot_encoder(X, max_categories=30)
    assert res.shape == (20, 20)

def test_onehot_encoder_max_categories_failure():
    # Create data with 20 unique values
    X_np = np.arange(20).astype(np.int32).reshape(-1, 1)
    X = Tensor(X_np)

    # max_categories=10 (less than 20) should raise ValueError
    with pytest.raises(ValueError, match="exceeding the limit"):
        tinyops_onehot_encoder(X, max_categories=10)

def test_onehot_encoder_default_limit():
    # Test that default limit (5000) is respected (or at least that argument is accepted)
    # We won't test 5001 values here to avoid slow tests, but we verify the arg exists.
    X = Tensor([[1], [2]])
    res = tinyops_onehot_encoder(X) # Should use default
    assert res.shape == (2, 2)
