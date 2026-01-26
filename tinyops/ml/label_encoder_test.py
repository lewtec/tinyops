import numpy as np
from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder
from tinygrad import Tensor

from tinyops._core import assert_close
from tinyops.ml.label_encoder import label_encoder as tinyops_label_encoder


def test_label_encoder():
    y_np = np.array([1, 5, 2, 8, 2, 5], dtype=np.int32)
    y_tiny = Tensor(y_np)

    # Sklearn's LabelEncoder
    sklearn_encoder = SklearnLabelEncoder()
    sklearn_result = sklearn_encoder.fit_transform(y_np)

    tinyops_result = tinyops_label_encoder(y_tiny)

    assert_close(tinyops_result, sklearn_result)


def test_label_encoder_string_equivalent():
    # Although tinygrad doesn't support strings, we can simulate
    # the behavior with large integer IDs.
    y_np = np.array([100, 200, 100, 300, 200], dtype=np.int32)
    y_tiny = Tensor(y_np)

    sklearn_encoder = SklearnLabelEncoder()
    sklearn_result = sklearn_encoder.fit_transform(y_np)

    tinyops_result = tinyops_label_encoder(y_tiny)

    assert_close(tinyops_result, sklearn_result)
