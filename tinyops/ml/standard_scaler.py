from tinygrad import Tensor


def standard_scaler(x: Tensor) -> Tensor:
    """
    Standardize features by removing the mean and scaling to unit variance.

    This implementation handles features with zero variance by leaving them unchanged
    (i.e., scaling by 1), matching the behavior of scikit-learn's StandardScaler.

    Args:
      x: The data to scale.

    Returns:
      The scaled data.
    """
    mean = x.mean(axis=0)
    # Use correction=0 for population standard deviation, matching sklearn
    std = x.std(axis=0, correction=0)

    # Avoid division by zero by setting scale to 1 where std is 0
    scale = Tensor.where(std == 0, 1.0, std)

    return (x - mean) / scale
