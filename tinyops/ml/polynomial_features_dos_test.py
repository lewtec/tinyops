import pytest
import math
from tinyops.ml.polynomial_features import polynomial_features

# Mock Tensor to avoid memory allocation
class MockTensor:
    def __init__(self, shape):
        self.shape = shape
        self.dtype = "float32"

def test_polynomial_features_dos():
    # n_features=100, degree=3
    # combinatorics:
    # d=1: comb(100, 1) = 100
    # d=2: comb(100+1, 2) = 101*100/2 = 5050
    # d=3: comb(100+2, 3) = 102*101*100/6 = 171700
    # Total > 100,000

    X_mock = MockTensor((10, 100))

    # We expect a ValueError because the output features would exceed the limit
    # Note: The limit is assumed to be 100,000 based on security guidelines.
    with pytest.raises(ValueError, match="exceeds the security limit"):
        polynomial_features(X_mock, degree=3, interaction_only=False)
