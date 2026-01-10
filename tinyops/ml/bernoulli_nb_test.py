import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.ml.bernoulli_nb import bernoulli_nb
from tinyops._core import assert_one_kernel

@pytest.mark.parametrize("n_samples, n_features, n_classes, alpha, binarize", [
    (150, 20, 4, 1.0, 0.5),
    (250, 30, 6, 0.5, 0.0),
    (200, 25, 5, 1.5, None),
])
def test_bernoulli_nb(n_samples, n_features, n_classes, alpha, binarize):
    # Generate synthetic data
    X_np, y_np = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=42
    )
    # For Bernoulli, data is often binary (0/1)
    if binarize is None:
        X_np = (X_np > 0.5).astype(np.float32)

    X_train_np, X_test_np, y_train_np, _ = train_test_split(
        X_np, y_np, test_size=0.25, random_state=42
    )

    X_train_np = X_train_np.astype(np.float32)
    X_test_np = X_test_np.astype(np.float32)

    # Convert to tinygrad Tensors and realize
    X_train = Tensor(X_train_np).realize()
    y_train = Tensor(y_train_np).realize()
    X_test = Tensor(X_test_np).realize()

    # Pre-compute classes for the test
    _classes = Tensor(np.unique(y_train_np)).realize()

    # Define the kernel execution function
    @assert_one_kernel
    def run_kernel():
        result = bernoulli_nb(X_train, y_train, X_test, alpha=alpha, binarize=binarize, _classes=_classes)
        result.realize()
        return result

    # Get the result from the tinyops implementation
    tinyops_predictions = run_kernel()

    # Get the result from the scikit-learn implementation
    bnb = BernoulliNB(alpha=alpha, binarize=binarize)
    bnb.fit(X_train_np, y_train_np)
    sklearn_predictions = bnb.predict(X_test_np)

    # Compare the results
    assert_close(tinyops_predictions, sklearn_predictions, atol=0, rtol=0)
