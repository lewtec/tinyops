from tinygrad import Tensor

from tinyops.ops.linear_algebra.cholesky_decomposition import cholesky_decomposition


def _compute_weights(
    dimension: int,
    lambda_parameter: float,
    spread: float,
    prior_knowledge: float,
    dtype,
    device,
) -> tuple[Tensor, Tensor]:
    """Compute mean and covariance weights."""
    weight_rest = 0.5 / (dimension + lambda_parameter)
    mean_weight_center = lambda_parameter / (dimension + lambda_parameter)
    covariance_weight_center = lambda_parameter / (dimension + lambda_parameter) + (1 - spread**2 + prior_knowledge)

    rest_weights = Tensor.full((2 * dimension,), weight_rest, dtype=dtype, device=device)
    mean_weight_center_tensor = Tensor([mean_weight_center], dtype=dtype, device=device)
    covariance_weight_center_tensor = Tensor(
        [covariance_weight_center], dtype=dtype, device=device
    )

    mean_weights = Tensor.cat(mean_weight_center_tensor, rest_weights, dim=0)
    covariance_weights = Tensor.cat(covariance_weight_center_tensor, rest_weights, dim=0)

    return mean_weights, covariance_weights


def merwe_scaled_sigma_points(
    state_mean: Tensor,
    covariance: Tensor,
    spread: float,
    prior_knowledge: float,
    secondary_scaling: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute Merwe Scaled Sigma Points for the Unscented Kalman Filter.

    Args:
        state_mean: Mean vector of the state distribution, shape (n,).
        covariance: Covariance matrix, shape (n, n).
        spread: Controls sigma point spread around the mean (alpha).
        prior_knowledge: Incorporates distribution knowledge (beta).
            For Gaussian distributions, 2 is optimal.
        secondary_scaling: Secondary scaling parameter (kappa).

    Returns:
        A tuple of (sigma_points, mean_weights, covariance_weights):
            - sigma_points: Shape (2n+1, n).
            - mean_weights: Shape (2n+1,).
            - covariance_weights: Shape (2n+1,).
    """
    dimension = state_mean.shape[0]

    if state_mean.ndim == 2 and state_mean.shape[1] == 1:
        state_mean = state_mean.squeeze(1)

    lambda_parameter = spread**2 * (dimension + secondary_scaling) - dimension
    scaled_covariance = covariance * (lambda_parameter + dimension)
    lower_triangular = cholesky_decomposition(scaled_covariance)
    upper_triangular = lower_triangular.T

    sigma_plus = state_mean + upper_triangular
    sigma_minus = state_mean - upper_triangular

    sigma_points = Tensor.cat(state_mean.unsqueeze(0), sigma_plus, sigma_minus, dim=0)

    mean_weights, covariance_weights = _compute_weights(
        dimension, lambda_parameter, spread, prior_knowledge, state_mean.dtype, state_mean.device
    )

    return sigma_points, mean_weights, covariance_weights
