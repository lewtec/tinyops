from tinygrad import Tensor


def logistic_regression(X: Tensor, y: Tensor, weights: Tensor, bias: Tensor, lr: float = 0.01) -> tuple[Tensor, Tensor]:
    """
    Performs a single gradient descent update for logistic regression.

    Args:
        X: Input features, shape (n_samples, n_features).
        y: Target values, shape (n_samples,).
        weights: Current weights, shape (n_features,).
        bias: Current bias, a scalar value.
        lr: Learning rate.

    Returns:
        A tuple containing the updated weights and bias.
    """
    # Calculate predictions using the sigmoid function
    logits = X.matmul(weights) + bias
    predictions = logits.sigmoid()

    # Calculate the error
    error = predictions - y

    # Calculate the gradients
    gradient_weights = X.T.matmul(error) / X.shape[0]
    gradient_bias = error.mean()

    # Update the weights and bias
    new_weights = weights - lr * gradient_weights
    new_bias = bias - lr * gradient_bias

    return new_weights, new_bias
