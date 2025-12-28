from tinygrad import Tensor
from tinyops.linalg import inv

def ridge(X: Tensor, y: Tensor, alpha: float = 1.0) -> Tensor:
    """
    Calculates the weights for Ridge regression.

    Args:
        X: Input features, shape (n_samples, n_features).
        y: Target values, shape (n_samples,).
        alpha: Regularization strength.

    Returns:
        The weights for the linear model, shape (n_features,).
    """
    X_b = Tensor.cat(X, Tensor.ones(X.shape[0], 1), dim=1)
    n_features_b = X_b.shape[1]

    # Identity matrix for regularization, with the last diagonal element as 0.
    # This prevents regularization of the bias term.
    I = Tensor.eye(n_features_b - 1).pad(((0, 1), (0, 1)))

    # Ridge regression formula
    A = X_b.T @ X_b + alpha * I
    b = X_b.T @ y

    weights = inv(A) @ b
    return weights
