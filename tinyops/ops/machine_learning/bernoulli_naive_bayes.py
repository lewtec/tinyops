import numpy as np
from tinygrad import Tensor, dtypes


def bernoulli_naive_bayes(
    training_features: Tensor,
    training_labels: Tensor,
    test_features: Tensor,
    smoothing: float = 1.0,
    binarize_threshold: float | None = 0.0,
    _classes: Tensor | None = None,
) -> Tensor:
    """Predict labels using a Bernoulli Naive Bayes classifier.

    Args:
        training_features: Training feature matrix.
        training_labels: Training labels.
        test_features: Test feature matrix.
        smoothing: Laplace smoothing parameter.
        binarize_threshold: If not None, binarize features at this threshold.
        _classes: Explicit class labels tensor (for testing).

    Returns:
        Predicted labels for test samples.
    """
    if _classes is None:
        classes = Tensor(np.unique(training_labels.numpy()), dtype=training_labels.dtype)
    else:
        classes = _classes

    class_indices = Tensor.arange(classes.shape[0])

    if binarize_threshold is not None:
        training_features = (training_features > binarize_threshold).cast(dtypes.float32)
        test_features = (test_features > binarize_threshold).cast(dtypes.float32)

    label_one_hot = (training_labels.reshape(-1, 1) == classes).cast(dtypes.float32)
    class_counts = label_one_hot.sum(0)
    log_class_priors = (class_counts / training_labels.shape[0]).log()

    feature_counts = label_one_hot.T @ training_features
    smoothed_counts = feature_counts + smoothing
    smoothed_class_counts = class_counts + 2 * smoothing

    log_feature_probs = smoothed_counts.log() - smoothed_class_counts.reshape(-1, 1).log()
    log_complement_probs = (
        (smoothed_class_counts.reshape(-1, 1) - smoothed_counts).log()
        - smoothed_class_counts.reshape(-1, 1).log()
    )

    joint_log_likelihood = test_features @ (log_feature_probs - log_complement_probs).T
    joint_log_likelihood += log_complement_probs.sum(1).reshape(1, -1)

    posteriors = joint_log_likelihood + log_class_priors.unsqueeze(0)
    predicted_indices = posteriors.argmax(1)

    predicted_one_hot = (class_indices.reshape(1, -1) == predicted_indices.reshape(-1, 1)).cast(dtypes.float32)
    predictions = (predicted_one_hot @ classes.cast(dtypes.float32).reshape(-1, 1)).flatten()
    return predictions.cast(classes.dtype)
