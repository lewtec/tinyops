from tinygrad import Tensor, dtypes

def adaboost_classifier(
    estimators_predictions: Tensor,
    estimator_weights: Tensor,
    classes: Tensor,
    learning_rate: float = 1.0,
) -> Tensor:
    """
    Performs the prediction step of a pre-trained AdaBoost classifier.

    This function is stateless and operates purely on tensor inputs, implementing the
    weighted majority vote based on the SAMME algorithm for multi-class classification.

    Args:
        estimators_predictions: Tensor of shape (n_estimators, n_samples) containing
            the class predictions from each weak learner.
        estimator_weights: Tensor of shape (n_estimators,) containing the weight
            (alpha) for each weak learner.
        classes: Tensor of shape (n_classes,) containing the unique class labels.
        learning_rate: The learning rate that scales the contribution of each classifier.
            Defaults to 1.0.

    Returns:
        A Tensor of shape (n_samples,) containing the final predicted class labels.
    """
    n_classes = classes.shape[0]
    n_estimators = estimators_predictions.shape[0]

    # One-hot encode the predictions from each estimator
    # Shape: (n_estimators, n_samples, n_classes)
    one_hot_preds = (
        estimators_predictions.unsqueeze(-1) == classes.reshape(1, 1, n_classes)
    ).cast(dtypes.float32)

    # Calculate the weighted vote for each class
    # Reshape weights for broadcasting: (n_estimators, 1, 1)
    # Sum over estimators to get total vote per class for each sample
    # Shape: (n_samples, n_classes)
    weighted_votes = (
        one_hot_preds * estimator_weights.reshape(n_estimators, 1, 1) * learning_rate
    ).sum(axis=0)

    # Get the index of the class with the highest vote for each sample
    # Shape: (n_samples,)
    final_pred_indices = weighted_votes.argmax(axis=1)

    # Gather the final class labels using the indices
    # This requires a gather operation. A simple way is to use one_hot and matmul.
    final_preds = (
        (final_pred_indices.unsqueeze(-1) == Tensor.arange(n_classes).reshape(1, n_classes))
        .cast(dtypes.float32)
        @ classes.unsqueeze(-1)
    ).flatten()

    return final_preds
