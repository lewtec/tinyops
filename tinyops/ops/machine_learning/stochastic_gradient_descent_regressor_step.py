from tinygrad import Tensor


def stochastic_gradient_descent_regressor_step(
    features: Tensor,
    target: Tensor,
    weights: Tensor,
    bias: Tensor,
    learning_rate: float = 0.01,
) -> tuple[Tensor, Tensor]:
    """Perform one SGD step for a linear regressor using squared error loss.

    Args:
        features: Single sample features (n_features,).
        target: Target value (scalar).
        weights: Current weight vector (n_features,).
        bias: Current bias scalar.
        learning_rate: Step size.

    Returns:
        Tuple of (updated_weights, updated_bias).
    """
    prediction = features.dot(weights) + bias
    error = prediction - target

    new_weights = weights - learning_rate * (features * error)
    new_bias = bias - learning_rate * error
    return new_weights, new_bias
