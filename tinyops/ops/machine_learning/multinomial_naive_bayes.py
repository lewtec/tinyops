from tinygrad import Tensor

from tinyops.ops.machine_learning._naive_bayes import (
    _class_labels_from_posterior_scores,
    _prepare_naive_bayes_training,
)


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
    classes, _class_counts, log_class_priors, feature_counts = _prepare_naive_bayes_training(
        training_features,
        training_labels,
        classes=_classes,
    )
    feature_count = training_features.shape[1]

    total_features_per_class = feature_counts.sum(1).reshape(-1, 1)
    log_feature_probabilities = (feature_counts + smoothing).log() - (
        total_features_per_class + smoothing * feature_count
    ).log()

    log_likelihoods = test_features @ log_feature_probabilities.T
    posteriors = log_likelihoods + log_class_priors.unsqueeze(0)
    return _class_labels_from_posterior_scores(posteriors, classes)
