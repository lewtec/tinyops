
import pytest
from tinygrad import Tensor
from tinyops.ml.polynomial_features import polynomial_features, MAX_OUTPUT_FEATURES

def test_polynomial_features_dos_prevention():
    """
    Test that polynomial_features raises ValueError when the expected number of
    output features exceeds the safety limit, preventing Denial of Service via memory exhaustion.
    """
    # Case 1: Interaction only = False
    # n_features=100, degree=10
    # math.comb(100 + 10 - 1, 10) = math.comb(109, 10) approx 4.2e13 > 100,000
    n_features = 100
    n_samples = 1
    degree = 10

    X = Tensor.ones(n_samples, n_features)

    with pytest.raises(ValueError, match=f"Number of output features .* exceeds limit .*"):
        polynomial_features(X, degree=degree, interaction_only=False)

    # Case 2: Interaction only = True
    # n_features=100, degree=5
    # math.comb(100, 5) = 75,287,520 > 100,000
    degree_interaction = 5
    with pytest.raises(ValueError, match=f"Number of output features .* exceeds limit .*"):
        polynomial_features(X, degree=degree_interaction, interaction_only=True)

def test_polynomial_features_boundary():
    """
    Test that it works when just below or at the limit (conceptually).
    Since exact limit testing is hard with integer discrete steps, we just ensure
    normal usage still works.
    """
    n_features = 5
    n_samples = 10
    degree = 2
    # math.comb(5+2-1, 2) = math.comb(6, 2) = 15 (plus bias and degree 1 terms)
    # Total features small, should pass.

    X = Tensor.ones(n_samples, n_features)
    res = polynomial_features(X, degree=degree)
    assert res.shape[0] == n_samples
