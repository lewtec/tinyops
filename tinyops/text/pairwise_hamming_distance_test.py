import numpy as np
from tinygrad import Tensor, dtypes
from tinyops.text.pairwise_hamming_distance import pairwise_hamming_distance, MAX_INTERMEDIATE_ELEMENTS
from tinyops._core import assert_close
from sklearn.metrics import pairwise_distances
import pytest

@pytest.mark.parametrize("data", [
    np.random.randint(0, 2, size=(5, 10))
])
def test_pairwise_hamming_distance(data):
    # sklearn implementation
    expected = pairwise_distances(data, metric='hamming')

    # tinyops implementation
    X = Tensor(data)
    result = pairwise_hamming_distance(X)

    # Assert that the result is close to the expected output.
    assert_close(result, Tensor(expected, dtype=dtypes.float32))

def test_pairwise_hamming_distance_security_check():
    # Construct a shape that will exceed the security limit
    n_samples = int((MAX_INTERMEDIATE_ELEMENTS / 100)**0.5) + 1
    n_features = 100
    shape = (n_samples, n_features)

    X = Tensor.zeros(shape)  # Content doesn't matter, only shape

    # Verify that the function raises a ValueError for oversized inputs
    with pytest.raises(ValueError, match="exceeding the security limit"):
        pairwise_hamming_distance(X)
