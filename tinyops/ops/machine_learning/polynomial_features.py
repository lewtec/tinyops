import itertools

from tinygrad import Tensor


def polynomial_features(
    features: Tensor,
    degree: int = 2,
    interaction_only: bool = False,
    include_bias: bool = True,
) -> Tensor:
    """Generate polynomial and interaction features.

    Args:
        features: Input tensor of shape (n_samples, n_features).
        degree: Maximum degree of polynomial features.
        interaction_only: If True, only include products of distinct features.
        include_bias: If True, include a column of ones.

    Returns:
        Feature matrix with polynomial combinations.
    """
    sample_count, feature_count = features.shape

    if degree == 0:
        if include_bias:
            return Tensor.ones(sample_count, 1, dtype=features.dtype)
        return Tensor.zeros(sample_count, 0, dtype=features.dtype)

    feature_indices = range(feature_count)
    combinations = []
    for current_degree in range(1, degree + 1):
        if interaction_only:
            combinations.extend(itertools.combinations(feature_indices, current_degree))
        else:
            combinations.extend(itertools.combinations_with_replacement(feature_indices, current_degree))

    output_columns = []
    if include_bias:
        output_columns.append(Tensor.ones(sample_count, 1, dtype=features.dtype))

    for combination in combinations:
        term = features[:, list(combination)].prod(axis=1)
        output_columns.append(term.unsqueeze(1))

    if not output_columns:
        return Tensor.zeros(sample_count, 0, dtype=features.dtype)

    return Tensor.cat(*output_columns, dim=1)
