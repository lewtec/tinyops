import numpy as np
from sklearn.datasets import make_regression
from sklearn.svm import LinearSVR
from tinygrad import Tensor

from tinyops._core import assert_close, assert_one_kernel
from tinyops.ml.linear_svr import linear_svr


def test_linear_svr():
    # 1. Train a model in scikit-learn
    X_np, y_np = make_regression(n_samples=100, n_features=10, random_state=42)
    X_np, y_np = X_np.astype(np.float32), y_np.astype(np.float32)

    model = LinearSVR(random_state=42, tol=1e-5)
    model.fit(X_np, y_np)

    # 2. Extract parameters and test data
    coef_np = model.coef_.astype(np.float32).reshape(1, -1)  # Ensure 2D
    intercept_np = model.intercept_.astype(np.float32)

    # 3. Convert to tinygrad tensors
    x = Tensor(X_np)
    coef = Tensor(coef_np)
    intercept = Tensor(intercept_np)
    x.realize(), coef.realize(), intercept.realize()

    # 4. Run tinyops implementation (ensuring one kernel)
    @assert_one_kernel
    def run_tinyops():
        result = linear_svr(x, coef, intercept)
        result.realize()
        return result

    result_to = run_tinyops()

    # 5. Get scikit-learn's prediction
    expected_np = model.predict(X_np)

    # 6. Assert parity
    assert_close(result_to, expected_np, atol=1e-4, rtol=1e-4)
