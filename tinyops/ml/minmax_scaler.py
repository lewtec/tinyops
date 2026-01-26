from tinygrad import Tensor


def minmax_scaler(x: Tensor, feature_range: tuple[int, int] = (0, 1)) -> Tensor:
    """
    Transforms features by scaling each feature to a given range.

    This transformer scales and translates each feature individually such
    that it is in the given range on the training set, e.g., between
    zero and one.

    Args:
      x: The data to scale.
      feature_range: The desired range of transformed data.

    Returns:
      The scaled data.
    """
    min_val, max_val = feature_range
    data_min = x.min(axis=0)
    data_max = x.max(axis=0)

    data_range = data_max - data_min
    scale = (max_val - min_val) / Tensor.where(data_range == 0, 1.0, data_range)

    return (x - data_min) * scale + min_val
