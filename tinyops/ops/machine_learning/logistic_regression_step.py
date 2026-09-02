from tinygrad import Tensor


def logistic_regression_step(
    features: Tensor,
    labels: Tensor,
    weights: Tensor,
    bias: Tensor,
    learning_rate: float = 0.01,
) -> tuple[Tensor, Tensor]:
    """Perform one gradient descent step for logistic regression.

    Args:
        features: Input features (n_samples, n_features).
        labels: Target binary labels (n_samples,).
        weights: Current weight vector (n_features,).
        bias: Current bias scalar.
        learning_rate: Step size for gradient descent.

    Returns:
        Tuple of (updated_weights, updated_bias).
    """
    logits = features.matmul(weights) + bias
    predictions = logits.sigmoid()
    error = predictions - labels

    weight_gradient = features.T.matmul(error) / features.shape[0]
    bias_gradient = error.mean()

    new_weights = weights - learning_rate * weight_gradient
    new_bias = bias - learning_rate * bias_gradient
    return new_weights, new_bias
