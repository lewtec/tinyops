from tinygrad import Tensor, dtypes


def adaboost_classifier(
    estimator_predictions: Tensor,
    estimator_weights: Tensor,
    classes: Tensor,
    learning_rate: float = 1.0,
) -> Tensor:
    """Predict labels using a pre-trained AdaBoost classifier (SAMME).

    Args:
        estimator_predictions: Predictions from each estimator
            (n_estimators, n_samples).
        estimator_weights: Weight (alpha) for each estimator (n_estimators,).
        classes: Unique class labels (n_classes,).
        learning_rate: Scales each estimator's contribution.

    Returns:
        Final predicted class labels (n_samples,).
    """
    class_count = classes.shape[0]
    estimator_count = estimator_predictions.shape[0]

    one_hot = (
        estimator_predictions.unsqueeze(-1) == classes.reshape(1, 1, class_count)
    ).cast(dtypes.float32)

    weighted_votes = (
        one_hot * estimator_weights.reshape(estimator_count, 1, 1) * learning_rate
    ).sum(axis=0)

    predicted_indices = weighted_votes.argmax(axis=1)

    index_one_hot = (
        predicted_indices.unsqueeze(-1) == Tensor.arange(class_count).reshape(1, class_count)
    ).cast(dtypes.float32)

    return (index_one_hot @ classes.unsqueeze(-1)).flatten()
