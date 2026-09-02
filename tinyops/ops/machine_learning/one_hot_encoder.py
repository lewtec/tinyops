from tinygrad import Tensor, dtypes

from tinyops.ops._tensor_utils import unique_sorted_values


def one_hot_encoder(features: Tensor) -> Tensor:
    """Encode categorical features as one-hot binary vectors.

    Args:
        features: Input tensor of shape (n_samples, n_features) or (n_samples,).

    Returns:
        Dense one-hot encoded tensor.
    """
    if features.ndim == 1:
        features = features.unsqueeze(1)

    encoded_columns = []
    for column_index in range(features.shape[1]):
        column = features[:, column_index]
        categories = Tensor(unique_sorted_values(column), requires_grad=False, device=features.device)
        one_hot = column.unsqueeze(1) == categories.unsqueeze(0)
        encoded_columns.append(one_hot)

    if not encoded_columns:
        return Tensor.zeros(features.shape[0], 0, dtype=dtypes.float32)

    return Tensor.cat(*encoded_columns, dim=1).cast(dtypes.float32)
