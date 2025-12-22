from tinygrad import Tensor

def sgd_regressor(X: Tensor, y: Tensor, weights: Tensor, bias: Tensor, lr: float = 0.01) -> tuple[Tensor, Tensor]:
    """
    Performs a single stochastic gradient descent update for a linear regressor.

    Args:
        X: Input features for a single sample, shape (n_features,).
        y: Target value for a single sample, scalar.
        weights: Current weights, shape (n_features,).
        bias: Current bias, a scalar value.
        lr: Learning rate.

    Returns:
        A tuple containing the updated weights and bias.
    """
    # Calculate the prediction and error
    prediction = X.dot(weights) + bias
    error = prediction - y

    # Gradients for squared error loss are (prediction - y) * X and (prediction - y)
    gradient_weights = X * error
    gradient_bias = error

    # Update weights and bias
    new_weights = weights - lr * gradient_weights
    new_bias = bias - lr * gradient_bias

    return new_weights, new_bias
