from tinygrad import Tensor
from tinyops.linalg import cholesky

def sigma_points(x: Tensor, P: Tensor, alpha: float, beta: float, kappa: float) -> tuple[Tensor, Tensor, Tensor]:
    """
    Computes Merwe Scaled Sigma Points and weights.

    Args:
        x: Mean vector of shape (n,).
        P: Covariance matrix of shape (n, n).
        alpha: Spread parameter.
        beta: Prior knowledge parameter.
        kappa: Scaling parameter.

    Returns:
        tuple: (sigmas, Wm, Wc)
            sigmas: Sigma points (2n+1, n).
            Wm: Weights for mean (2n+1,).
            Wc: Weights for covariance (2n+1,).
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

    rest = Tensor.full((2*n,), c, dtype=x.dtype, device=x.device)

    Wm_0_t = Tensor([Wm_0], dtype=x.dtype, device=x.device)
    Wc_0_t = Tensor([Wc_0], dtype=x.dtype, device=x.device)

    Wm = Tensor.cat(Wm_0_t, rest, dim=0)
    Wc = Tensor.cat(Wc_0_t, rest, dim=0)

    return sigmas, Wm, Wc
