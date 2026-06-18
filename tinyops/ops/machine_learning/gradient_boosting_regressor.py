from tinygrad import Tensor

from tinyops.ops.machine_learning.decision_tree_regressor import decision_tree_regressor


def gradient_boosting_regressor(
    samples: Tensor,
    estimators: list[dict],
    learning_rate: float,
    initial_prediction: Tensor,
) -> Tensor:
    """Predict regression targets using a pre-trained gradient boosting model.

    Args:
        samples: Input samples (n_samples, n_features).
        estimators: List of decision tree dictionaries (boosting stages).
        learning_rate: Learning rate for each stage.
        initial_prediction: Initial prediction (usually mean of training targets).

    Returns:
        Predicted values (n_samples,).
    """
    prediction = initial_prediction.expand(samples.shape[0])
    for tree in estimators:
        prediction = prediction + decision_tree_regressor(samples, tree) * learning_rate
    return prediction
