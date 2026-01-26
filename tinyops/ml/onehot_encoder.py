import numpy as np
from tinygrad import Tensor, dtypes


def onehot_encoder(X: Tensor) -> Tensor:
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
