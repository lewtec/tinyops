from tinygrad import Tensor

from tinyops.ops.machine_learning._scaling import replace_zero_scale_with_one


def max_absolute_scaler(features: Tensor) -> Tensor:
    """Scale each feature by its maximum absolute value.

    Does not shift the data, preserving sparsity.

    Args:
        features: Input tensor of shape (n_samples, n_features).

    Returns:
        Scaled tensor with maximum absolute value of 1 per feature.
    """
    max_absolute = features.abs().max(axis=0)
    safe_scale = replace_zero_scale_with_one(max_absolute)
    return features / safe_scale
