from tinygrad import Tensor

from tinyops.ops.machine_learning._scaling import replace_zero_scale_with_one


def standard_scaler(features: Tensor) -> Tensor:
    """Standardize features by removing the mean and scaling to unit variance.

    Features with zero variance are left unchanged.

    Args:
        features: Input tensor of shape (n_samples, n_features).

    Returns:
        Standardized tensor.
    """
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0, correction=0)
    safe_scale = replace_zero_scale_with_one(feature_std)
    return (features - feature_mean) / safe_scale
