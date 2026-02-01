from tinygrad import Tensor

from tinyops.linalg import cholesky


def sigma_points(x: Tensor, P: Tensor, alpha: float, beta: float, kappa: float) -> tuple[Tensor, Tensor, Tensor]:
    """
    Computes Merwe Scaled Sigma Points and weights for the Unscented Kalman Filter (UKF).

    Sigma points are a set of deterministic vectors chosen to capture the mean and covariance
    of a random variable. Merwe Scaled Sigma Points allow for precise control over the spread
    of the points, improving stability and accuracy for non-linear transformations.

    Args:
        x: Mean vector of state distribution, shape (n,).
        P: Covariance matrix of state distribution, shape (n, n).
        alpha: Spread parameter. Determines the spread of the sigma points around the mean.
            Typically a small positive value (e.g., 1e-3).
        beta: Prior knowledge parameter. Used to incorporate prior knowledge of the distribution.
            For Gaussian distributions, beta = 2 is optimal.
        kappa: Secondary scaling parameter. Usually set to 0 or 3 - n.

    Returns:
        A tuple of (sigmas, Wm, Wc):
            - sigmas: The generated sigma points with shape (2n+1, n).
            - Wm: Weights for the mean calculation, shape (2n+1,).
            - Wc: Weights for the covariance calculation, shape (2n+1,).
    """
    n = x.shape[0]

    # Ensure x is (n,)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x.squeeze(1)

    lambda_ = alpha**2 * (n + kappa) - n

    # P_scaled = (lambda_ + n) * P
    P_scaled = P * (lambda_ + n)

    # Cholesky decomposition L s.t. L @ L.T = P_scaled (Lower triangular)
    L = cholesky(P_scaled)

    # We use U = L.T (Upper triangular)
    # sigma_points logic adds rows of U (which are columns of L)
    U = L.T

    # x broadcasted to (n, n)
    sigmas_plus = x + U
    sigmas_minus = x - U

    sigmas = Tensor.cat(x.unsqueeze(0), sigmas_plus, sigmas_minus, dim=0)

    # Weights
    c = 0.5 / (n + lambda_)

    Wm_0 = lambda_ / (n + lambda_)
    Wc_0 = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)

    rest = Tensor.full((2 * n,), c, dtype=x.dtype, device=x.device)

    Wm_0_t = Tensor([Wm_0], dtype=x.dtype, device=x.device)
    Wc_0_t = Tensor([Wc_0], dtype=x.dtype, device=x.device)

    Wm = Tensor.cat(Wm_0_t, rest, dim=0)
    Wc = Tensor.cat(Wc_0_t, rest, dim=0)

    return sigmas, Wm, Wc
