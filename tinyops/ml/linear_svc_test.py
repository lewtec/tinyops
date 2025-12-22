import pytest
import numpy as np
from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.ml.linear_svc import linear_svc
from tinyops.test_utils import assert_one_kernel

from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification

@pytest.mark.parametrize("n_classes", [2, 4])
def test_linear_svc(n_classes):
    # 1. Train a model in scikit-learn
    X_np, y_np = make_classification(n_samples=100, n_features=10, n_classes=n_classes, n_informative=5, random_state=42)
    X_np = X_np.astype(np.float32)

    model = LinearSVC(random_state=42)
    model.fit(X_np, y_np)

    # 2. Extract parameters and test data
    coef_np = model.coef_.astype(np.float32)
    intercept_np = model.intercept_.astype(np.float32)

    # 3. Convert to tinygrad tensors
    x = Tensor(X_np)
    coef = Tensor(coef_np)
    intercept = Tensor(intercept_np)
    x.realize(), coef.realize(), intercept.realize()

    # 4. Run tinyops implementation (ensuring one kernel)
    @assert_one_kernel
    def run_tinyops():
        result = linear_svc(x, coef, intercept)
        result.realize()
        return result

    result_to = run_tinyops()

    # 5. Get scikit-learn's prediction
    expected_np = model.decision_function(X_np)

    # 6. Assert parity
    # Sklearn returns a 1D array for binary classification, so we reshape to match
    if n_classes == 2:
      expected_np = expected_np.reshape(-1, 1)

    assert_close(result_to, expected_np, atol=1e-5, rtol=1e-5)
