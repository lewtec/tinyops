import itertools
import math

from tinygrad import Tensor

MAX_OUTPUT_FEATURES = 1_000_000


def polynomial_features(
    X: Tensor, degree: int = 2, interaction_only: bool = False, include_bias: bool = True
) -> Tensor:
    """
    Generate polynomial and interaction features.

    Generate a new feature matrix consisting of all polynomial combinations of the features
    with degree less than or equal to the specified degree. For example, if an input sample is
    two dimensional and of the form [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].

    Args:
        X: Input tensor of shape (n_samples, n_features).
        degree: The degree of the polynomial features. Default = 2.
        interaction_only: If true, only interaction features are produced: features that are
            products of at most ``degree`` *distinct* input features (so not ``x[1] ** 2``,
            ``x[0] * x[2] ** 3``, etc.).
        include_bias: If True (default), then include a bias column, the feature in which
            all polynomial powers are zero (i.e. a column of ones - acts as an intercept term in a linear model).

    Returns:
        The new feature matrix, shape (n_samples, n_output_features).

    Warning:
        Be aware that the number of output features scales exponentially with `degree` and `n_features`.
        High values can lead to massive memory consumption and Denial of Service (DoS).
    """
    n_samples, n_features = X.shape

    if degree == 0:
        return Tensor.ones(n_samples, 1, dtype=X.dtype) if include_bias else Tensor.zeros(n_samples, 0, dtype=X.dtype)

    expected_features = 1 if include_bias else 0
    for d in range(1, degree + 1):
        if interaction_only:
            expected_features += math.comb(n_features, d)
        else:
            expected_features += math.comb(n_features + d - 1, d)

        if expected_features > MAX_OUTPUT_FEATURES:
            raise ValueError(
                f"Too many output features (expected > {MAX_OUTPUT_FEATURES}). "
                "Decrease degree or n_features to avoid Denial of Service."
            )

    feature_indices = range(n_features)

    # Generate combinations of feature indices. This part is orchestration and runs on the CPU.
    combs = []
    # The loop generates combinations for each degree from 1 up to the specified degree.
    for d in range(1, degree + 1):
        if interaction_only:
            combs.extend(itertools.combinations(feature_indices, d))
        else:
            combs.extend(itertools.combinations_with_replacement(feature_indices, d))

    # The final features are constructed from these combinations.
    # Scikit-learn's default order is [1, X_features, higher_degree_features...].
    # The combinations are generated in increasing order of degree, so this order is preserved.

    output_features = []
    if include_bias:
        output_features.append(Tensor.ones(n_samples, 1, dtype=X.dtype))

    # Compute the product of features for each combination in a vectorized manner.
    for comb in combs:
        # Select the feature columns for the current combination and compute their product.
        # This is more efficient than multiplying them one by one in a Python loop.
        term = X[:, list(comb)].prod(axis=1)
        output_features.append(term.unsqueeze(1))

    if not output_features:
        return Tensor.zeros(n_samples, 0, dtype=X.dtype)

    return Tensor.cat(*output_features, dim=1)
