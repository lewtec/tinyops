from tinygrad import Tensor
from tinyops.ml.decision_tree_regressor import decision_tree_regressor

def gradient_boosting_regressor(X: Tensor, estimators: list[dict], learning_rate: float, init_prediction: Tensor) -> Tensor:
    """
    Predicts regression target for X using a pre-trained gradient boosting model.

    Args:
        X: Input samples of shape (n_samples, n_features).
        estimators: A list of dictionaries, where each dictionary represents a trained decision tree estimator.
        learning_rate: The learning rate.
        init_prediction: The initial prediction, usually the mean of the training target values.

    Returns:
        A Tensor of shape (n_samples,) with the predicted regression values.
    """
    y_pred = init_prediction.expand(X.shape[0])
    for tree in estimators:
        y_pred = y_pred + decision_tree_regressor(X, tree) * learning_rate
    return y_pred
