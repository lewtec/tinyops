import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.ml.multinomial_nb import multinomial_nb
from tinyops._core import assert_one_kernel

@pytest.mark.parametrize("n_samples, n_features, n_classes, alpha", [
    (150, 20, 4, 1.0),
    (250, 30, 6, 0.5),
])
def test_multinomial_nb(n_samples, n_features, n_classes, alpha):
    # Generate synthetic data with non-negative integer features
    X_np, y_np = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=42
    )
    X_np = np.abs(X_np * 100).astype(np.int32).astype(np.float32)

    X_train_np, X_test_np, y_train_np, _ = train_test_split(
        X_np, y_np, test_size=0.25, random_state=42
    )

    # Convert to tinygrad Tensors and realize
    X_train = Tensor(X_train_np).realize()
    y_train = Tensor(y_train_np).realize()
    X_test = Tensor(X_test_np).realize()

    # Pre-compute classes for the test
    _classes = Tensor(np.unique(y_train_np)).realize()

    # Define the kernel execution function
    @assert_one_kernel
    def run_kernel():
        result = multinomial_nb(X_train, y_train, X_test, alpha=alpha, _classes=_classes)
        result.realize()
        return result

    # Get the result from the tinyops implementation
    tinyops_predictions = run_kernel()

    # Get the result from the scikit-learn implementation
    mnb = MultinomialNB(alpha=alpha)
    mnb.fit(X_train_np, y_train_np)
    sklearn_predictions = mnb.predict(X_test_np)

    # Compare the results
    assert_close(tinyops_predictions, sklearn_predictions, atol=0, rtol=0)
