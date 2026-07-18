from tinygrad import Tensor

from tinyops.ops.machine_learning._scaling import replace_zero_scale_with_one


def min_max_scaler(
    features: Tensor,
    target_range: tuple[float, float] = (0, 1),
) -> Tensor:
    """Scale features to a target range using min-max normalization.

    Args:
        features: Input tensor of shape (n_samples, n_features).
        target_range: Desired (min, max) of the transformed data.

    Returns:
        Scaled tensor.
    """
    range_min, range_max = target_range
    data_min = features.min(axis=0)
    data_max = features.max(axis=0)
    data_range = data_max - data_min
    safe_range = replace_zero_scale_with_one(data_range)
    scale = (range_max - range_min) / safe_range
    return (features - data_min) * scale + range_min
