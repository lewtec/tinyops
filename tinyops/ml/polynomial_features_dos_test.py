
import pytest
from tinygrad import Tensor
from tinyops.ml.polynomial_features import polynomial_features

def test_dos_protection():
    # n_features=20, degree=6
    # comb(20+6-1, 6) = comb(25, 6) = 177,100
    # This exceeds the safety threshold of 100,000.
    X = Tensor.zeros(1, 20)
    with pytest.raises(ValueError, match="Too many output features"):
        polynomial_features(X, degree=6)
