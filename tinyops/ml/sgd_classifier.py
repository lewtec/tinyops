from tinygrad import Tensor


def sgd_classifier(X: Tensor, y: Tensor, weights: Tensor, bias: Tensor, lr: float = 0.01) -> tuple[Tensor, Tensor]:
    """
    Performs a single stochastic gradient descent update for a linear classifier using the hinge loss.

    Args:
        X: Input features for a single sample, shape (n_features,).
        y: Target value for a single sample (must be -1 or 1).
        weights: Current weights, shape (n_features,).
        bias: Current bias, a scalar value.
        lr: Learning rate.

    Returns:
        A tuple containing the updated weights and bias.
    """
    # Calculate the decision function
    decision = X.dot(weights) + bias

    # Hinge loss gradient calculation
    condition = (y * decision) < 1
    gradient_multiplier = Tensor.where(condition, -y, 0.0)

    # Calculate gradients
    gradient_weights = X * gradient_multiplier
    gradient_bias = gradient_multiplier

    # Update weights and bias
    new_weights = weights - lr * gradient_weights
    new_bias = bias - lr * gradient_bias

    return new_weights, new_bias
