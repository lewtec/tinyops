import pytest
from tinygrad import Tensor
from tinyops.ml.polynomial_features import polynomial_features

def test_polynomial_features_dos_prevention():
    # n_features=50, degree=4 -> ~292k features.
    # This should be blocked by the 100k limit.
    # Formula: sum(comb(50+d-1, d) for d in 1..4) + 1 (bias)
    # d=1: 50
    # d=2: 1275
    # d=3: 22100
    # d=4: 292825
    # Total > 300k
    n_features = 50
    degree = 4
    X = Tensor.zeros(1, n_features)

    # We expect a ValueError to be raised due to the new limit.
    with pytest.raises(ValueError, match="Too many output features"):
        polynomial_features(X, degree=degree)
