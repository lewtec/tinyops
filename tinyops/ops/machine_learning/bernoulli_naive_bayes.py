from tinygrad import Tensor, dtypes

from tinyops.ops.machine_learning._naive_bayes import (
    BERNOULLI_OUTCOME_COUNT,
    _class_labels_from_posterior_scores,
    _prepare_naive_bayes_training,
)


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
    if binarize_threshold is not None:
        training_features = (training_features > binarize_threshold).cast(dtypes.float32)
        test_features = (test_features > binarize_threshold).cast(dtypes.float32)

    classes, class_counts, log_class_priors, feature_counts = _prepare_naive_bayes_training(
        training_features,
        training_labels,
        classes=_classes,
    )

    smoothed_counts = feature_counts + smoothing
    smoothed_class_counts = class_counts + BERNOULLI_OUTCOME_COUNT * smoothing

    log_feature_probs = smoothed_counts.log() - smoothed_class_counts.reshape(-1, 1).log()
    log_complement_probs = (
        smoothed_class_counts.reshape(-1, 1) - smoothed_counts
    ).log() - smoothed_class_counts.reshape(-1, 1).log()

    joint_log_likelihood = test_features @ (log_feature_probs - log_complement_probs).T
    joint_log_likelihood += log_complement_probs.sum(1).reshape(1, -1)

    posteriors = joint_log_likelihood + log_class_priors.unsqueeze(0)
    return _class_labels_from_posterior_scores(posteriors, classes)
