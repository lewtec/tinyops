import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.neighbors import NearestNeighbors as SklearnNearestNeighbors
from tinygrad import Tensor

from tinyops.ml.nearest_neighbors import nearest_neighbors


@pytest.mark.parametrize(
    "shape, n_neighbors",
    [
        ((10, 3), 3),
        ((50, 5), 5),
        ((100, 2), 10),
    ],
)
def test_nearest_neighbors(shape, n_neighbors):
    X_np = np.random.rand(*shape).astype(np.float32)
    X = Tensor(X_np).realize()

    # Run without the kernel assertion to debug the core logic first.
    tinyops_result = nearest_neighbors(X, n_neighbors)
    tinyops_result.realize()

    nn = SklearnNearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X_np)
    _, sklearn_indices = nn.kneighbors(X_np)

    # Sort both results to ensure they are comparable, as the order of equally distant neighbors can differ.
    tinyops_sorted_indices = np.sort(tinyops_result.numpy(), axis=1)
    sklearn_sorted_indices = np.sort(sklearn_indices, axis=1)

    assert_array_equal(tinyops_sorted_indices, sklearn_sorted_indices)
