from tinygrad import Tensor


def max_absolute_scaler(features: Tensor) -> Tensor:
    """Scale each feature by its maximum absolute value.

    Does not shift the data, preserving sparsity.

    Args:
        features: Input tensor of shape (n_samples, n_features).

    Returns:
        Scaled tensor with maximum absolute value of 1 per feature.
    """
    max_absolute = features.abs().max(axis=0)
    safe_scale = Tensor.where(max_absolute == 0, 1.0, max_absolute)
    return features / safe_scale
