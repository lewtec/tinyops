from tinygrad import Tensor

from tinyops.ops.machine_learning._scaling import replace_zero_scale_with_one
from tinyops.ops.statistics.percentile import percentile


def robust_scaler(features: Tensor) -> Tensor:
    """Scale features using statistics robust to outliers.

    Centers on the median and scales by the interquartile range (IQR).

    Args:
        features: Input tensor of shape (n_samples, n_features).

    Returns:
        Robustly scaled tensor.
    """
    feature_median = percentile(features, 50, axis=0)
    first_quartile = percentile(features, 25, axis=0)
    third_quartile = percentile(features, 75, axis=0)
    interquartile_range = third_quartile - first_quartile
    safe_scale = replace_zero_scale_with_one(interquartile_range)
    return (features - feature_median) / safe_scale
