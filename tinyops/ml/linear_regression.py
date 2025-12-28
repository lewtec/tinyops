from tinygrad import Tensor
from tinyops.linalg import pinv

def linear_regression(X: Tensor, y: Tensor) -> Tensor:
    """
    Calculates the weights for linear regression using the normal equation.

    Args:
        X: Input features, shape (n_samples, n_features).
        y: Target values, shape (n_samples,).

    Returns:
        The weights for the linear model, shape (n_features,).
    """
    X_b = Tensor.cat(X, Tensor.ones(X.shape[0], 1), dim=1)
    weights = pinv(X_b.T @ X_b) @ X_b.T @ y
    return weights
