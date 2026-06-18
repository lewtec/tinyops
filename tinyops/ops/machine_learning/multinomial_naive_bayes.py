import numpy as np
from tinygrad import Tensor, dtypes


def multinomial_naive_bayes(
    training_features: Tensor,
    training_labels: Tensor,
    test_features: Tensor,
    smoothing: float = 1.0,
    _classes: Tensor | None = None,
) -> Tensor:
    """Predict labels using a Multinomial Naive Bayes classifier.

    Args:
        training_features: Training feature matrix (n_train, n_features).
        training_labels: Training labels (n_train,).
        test_features: Test feature matrix (n_test, n_features).
        smoothing: Laplace smoothing parameter.
        _classes: Explicit class labels tensor (for testing).

    Returns:
        Predicted labels for test samples.
    """
    if _classes is None:
        classes = Tensor(np.unique(training_labels.numpy()), dtype=training_labels.dtype)
    else:
        classes = _classes

    class_indices = Tensor.arange(classes.shape[0])
    feature_count = training_features.shape[1]

    label_one_hot = (training_labels.reshape(-1, 1) == classes).cast(dtypes.float32)
    class_counts = label_one_hot.sum(0)
    log_class_priors = (class_counts / training_labels.shape[0]).log()

    feature_counts = label_one_hot.T @ training_features
    total_features_per_class = feature_counts.sum(1).reshape(-1, 1)
    log_feature_probabilities = (
        (feature_counts + smoothing).log()
        - (total_features_per_class + smoothing * feature_count).log()
    )

    log_likelihoods = test_features @ log_feature_probabilities.T
    posteriors = log_likelihoods + log_class_priors.unsqueeze(0)
    predicted_indices = posteriors.argmax(1)

    predicted_one_hot = (class_indices.reshape(1, -1) == predicted_indices.reshape(-1, 1)).cast(dtypes.float32)
    predictions = (predicted_one_hot @ classes.cast(dtypes.float32).reshape(-1, 1)).flatten()
    return predictions.cast(classes.dtype)
