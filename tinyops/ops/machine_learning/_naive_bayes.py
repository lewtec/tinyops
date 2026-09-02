"""Shared helpers for naive Bayes classifiers."""

from tinygrad import Tensor, dtypes

from tinyops.ops._tensor_utils import unique_sorted_values

# Bernoulli features are binary (present / absent); Laplace smoothing adds one
# pseudo-count for each outcome when estimating P(feature | class).
BERNOULLI_OUTCOME_COUNT = 2


def _prepare_naive_bayes_training(
    training_features: Tensor,
    training_labels: Tensor,
    classes: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Resolve class labels and shared training aggregates.

    Returns:
        classes: Sorted unique class labels.
        class_counts: Sample count per class.
        log_class_priors: Natural log of class prior probabilities.
        feature_counts: Per-class feature sums (n_classes, n_features).
    """
    if classes is None:
        classes = Tensor(unique_sorted_values(training_labels), dtype=training_labels.dtype)

    label_one_hot = (training_labels.reshape(-1, 1) == classes).cast(dtypes.float32)
    class_counts = label_one_hot.sum(0)
    log_class_priors = (class_counts / training_labels.shape[0]).log()
    feature_counts = label_one_hot.T @ training_features
    return classes, class_counts, log_class_priors, feature_counts


def _class_labels_from_posterior_scores(posteriors: Tensor, classes: Tensor) -> Tensor:
    """Map argmax class indices from posterior scores back to label values."""
    predicted_indices = posteriors.argmax(1)
    class_indices = Tensor.arange(classes.shape[0])
    predicted_one_hot = (class_indices.reshape(1, -1) == predicted_indices.reshape(-1, 1)).cast(dtypes.float32)
    predictions = (predicted_one_hot @ classes.cast(dtypes.float32).reshape(-1, 1)).flatten()
    return predictions.cast(classes.dtype)
