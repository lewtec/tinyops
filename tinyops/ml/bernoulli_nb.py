import numpy as np
from tinygrad import Tensor, dtypes


def bernoulli_nb(
    X_train: Tensor,
    y_train: Tensor,
    X_test: Tensor,
    alpha: float = 1.0,
    binarize: float | None = 0.0,
    _classes: Tensor | None = None,
) -> Tensor:
    if _classes is None:
        y_np = y_train.numpy()
        classes_np = np.unique(y_np)
        classes = Tensor(classes_np, dtype=y_train.dtype)
    else:
        classes = _classes

    class_indices = Tensor.arange(classes.shape[0])

    if binarize is not None:
        X_train = (X_train > binarize).cast(dtypes.float32)
        X_test = (X_test > binarize).cast(dtypes.float32)

    y_one_hot = (y_train.reshape(-1, 1) == classes).cast(dtypes.float32)

    class_counts = y_one_hot.sum(0)
    log_class_priors = (class_counts / y_train.shape[0]).log()

    feature_counts = y_one_hot.T @ X_train

    smoothed_counts = feature_counts + alpha
    smoothed_class_counts = class_counts + 2 * alpha

    log_feature_probs = smoothed_counts.log() - smoothed_class_counts.reshape(-1, 1).log()

    neg_log_feature_probs = (
        smoothed_class_counts.reshape(-1, 1) - smoothed_counts
    ).log() - smoothed_class_counts.reshape(-1, 1).log()

    jll = X_test @ (log_feature_probs - neg_log_feature_probs).T
    jll += neg_log_feature_probs.sum(1).reshape(1, -1)

    posteriors = jll + log_class_priors.unsqueeze(0)

    pred_indices = posteriors.argmax(1)

    pred_one_hot = (class_indices.reshape(1, -1) == pred_indices.reshape(-1, 1)).cast(dtypes.float32)
    predictions = (pred_one_hot @ classes.cast(dtypes.float32).reshape(-1, 1)).flatten()

    return predictions.cast(classes.dtype)
