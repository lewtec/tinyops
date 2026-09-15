from tinygrad import Tensor


def stochastic_gradient_descent_classifier_step(
    features: Tensor,
    label: Tensor,
    weights: Tensor,
    bias: Tensor,
    learning_rate: float = 0.01,
) -> tuple[Tensor, Tensor]:
    """Perform one SGD step for a linear classifier using hinge loss.

    Args:
        features: Single sample features (n_features,).
        label: Target value (-1 or 1).
        weights: Current weight vector (n_features,).
        bias: Current bias scalar.
        learning_rate: Step size.

    Returns:
        Tuple of (updated_weights, updated_bias).
    """
    decision = features.dot(weights) + bias
    is_misclassified = (label * decision) < 1
    gradient_multiplier = Tensor.where(is_misclassified, -label, 0.0)

    new_weights = weights - learning_rate * (features * gradient_multiplier)
    new_bias = bias - learning_rate * gradient_multiplier
    return new_weights, new_bias
