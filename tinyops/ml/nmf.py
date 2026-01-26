from tinygrad import Tensor


def nmf(X: Tensor, n_components: int, max_iter: int = 200, tol: float = 1e-4) -> tuple[Tensor, Tensor]:
    """
    Non-Negative Matrix Factorization (NMF).

    Finds two non-negative matrices (W, H) whose product approximates the non-negative matrix X.
    This implementation uses the multiplicative update algorithm.

    Args:
        X: Input tensor of shape (n_samples, n_features). All values must be non-negative.
        n_components: Number of components to extract.
        max_iter: Maximum number of iterations.
        tol: Tolerance of the stopping condition.

    Returns:
        A tuple containing:
        - W: Non-negative matrix of shape (n_samples, n_components).
        - H: Non-negative matrix of shape (n_components, n_features).
    """
    n_samples, n_features = X.shape

    # Initialize W and H with non-negative random values
    W = Tensor.rand(n_samples, n_components)
    H = Tensor.rand(n_components, n_features)

    # Small constant to avoid division by zero
    epsilon = 1e-7
    prev_error = float("inf")

    for _ in range(max_iter):
        # Update H
        numerator_h = W.transpose() @ X
        denominator_h = W.transpose() @ W @ H + epsilon
        H = H * numerator_h / denominator_h

        # Update W
        numerator_w = X @ H.transpose()
        denominator_w = W @ H @ H.transpose() + epsilon
        W = W * numerator_w / denominator_w

        # Check for convergence based on the change in error
        error_tensor = (X - W @ H).pow(2).sum().sqrt()
        current_error = error_tensor.item()
        if abs(prev_error - current_error) < tol:
            break
        prev_error = current_error

    return W, H
