import numpy as np
import pytest
from sklearn.metrics import pairwise_distances
from tinygrad import Tensor, dtypes

from tinyops._core import assert_close
from tinyops.text.pairwise_hamming_distance import pairwise_hamming_distance


@pytest.mark.parametrize("data", [np.random.randint(0, 2, size=(5, 10))])
def test_pairwise_hamming_distance(data):
    # sklearn implementation
    expected = pairwise_distances(data, metric="hamming")

    # tinyops implementation
    X = Tensor(data)
    result = pairwise_hamming_distance(X)

    # Assert that the result is close to the expected output.
    assert_close(result, Tensor(expected, dtype=dtypes.float32))
