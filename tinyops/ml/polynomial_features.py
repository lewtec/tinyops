from tinygrad import Tensor
import itertools
import math

MAX_OUTPUT_FEATURES = 100000

def polynomial_features(X: Tensor, degree: int = 2, interaction_only: bool = False, include_bias: bool = True) -> Tensor:
    n_samples, n_features = X.shape

    if degree == 0:
        return Tensor.ones(n_samples, 1, dtype=X.dtype) if include_bias else Tensor.zeros(n_samples, 0, dtype=X.dtype)

    # Validate output size to prevent DoS
    n_output_features = 1 if include_bias else 0
    for d in range(1, degree + 1):
        if interaction_only:
            n_output_features += math.comb(n_features, d)
        else:
            n_output_features += math.comb(n_features + d - 1, d)

    if n_output_features > MAX_OUTPUT_FEATURES:
        raise ValueError(f"Number of output features {n_output_features} exceeds limit {MAX_OUTPUT_FEATURES}")

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
