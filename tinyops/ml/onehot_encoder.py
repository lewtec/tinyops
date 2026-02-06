from tinygrad import Tensor, dtypes


def onehot_encoder(X: Tensor) -> Tensor:
    """
    Encode categorical features as a one-hot numeric array.

    This function performs both "fit" and "transform" in a single step. It identifies
    unique values in each column (feature) and transforms them into binary vectors.

    The logic is analogous to `sklearn.preprocessing.OneHotEncoder`, but simplified for
    stateless tensor operations.

    Args:
        X: Input tensor of shape (n_samples, n_features) or (n_samples,).

    Returns:
        A dense tensor of shape (n_samples, n_encoded_features).
        Each feature from input is expanded into `n_categories` binary columns.

    Warning:
        This function returns a dense Tensor. If the number of categories is very large,
        memory consumption will be high.
    """
    import numpy as np

    if X.ndim == 1:
        X = X.unsqueeze(1)

    encoded_cols = []
    for i in range(X.shape[1]):
        col = X[:, i]
        # "fit" step: find unique categories. This is analogous to sklearn's fit method.
        # Using numpy here is a pragmatic choice as a pure-tensor unique is complex.
        categories_np = np.unique(col.numpy())
        categories = Tensor(categories_np, requires_grad=False, device=X.device)

        # "transform" step: create one-hot encoding based on discovered categories.
        # This uses broadcasting: (N, 1) == (1, C) -> (N, C)
        one_hot_col = col.unsqueeze(1) == categories.unsqueeze(0)
        encoded_cols.append(one_hot_col)

    if not encoded_cols:
        return Tensor.zeros(X.shape[0], 0, dtype=dtypes.float32)

    return Tensor.cat(*encoded_cols, dim=1).cast(dtypes.float32)
