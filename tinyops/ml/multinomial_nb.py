import numpy as np
from tinygrad import Tensor, dtypes


def multinomial_nb(
    X_train: Tensor, y_train: Tensor, X_test: Tensor, alpha: float = 1.0, _classes: Tensor | None = None
) -> Tensor:
    if _classes is None:
        y_np = y_train.numpy()
        classes_np = np.unique(y_np)
        classes = Tensor(classes_np, dtype=y_train.dtype)
    else:
        classes = _classes

    class_indices = Tensor.arange(classes.shape[0])
    n_features = X_train.shape[1]

    y_one_hot = (y_train.reshape(-1, 1) == classes).cast(dtypes.float32)

    class_counts = y_one_hot.sum(0)
    log_class_priors = (class_counts / y_train.shape[0]).log()

    feature_counts = y_one_hot.T @ X_train
    total_features_per_class = feature_counts.sum(1).reshape(-1, 1)

    log_feature_probs = (feature_counts + alpha).log() - (total_features_per_class + alpha * n_features).log()

    log_likelihoods = X_test @ log_feature_probs.T

    posteriors = log_likelihoods + log_class_priors.unsqueeze(0)

    pred_indices = posteriors.argmax(1)

    pred_one_hot = (class_indices.reshape(1, -1) == pred_indices.reshape(-1, 1)).cast(dtypes.float32)
    predictions = (pred_one_hot @ classes.cast(dtypes.float32).reshape(-1, 1)).flatten()

    return predictions.cast(classes.dtype)
