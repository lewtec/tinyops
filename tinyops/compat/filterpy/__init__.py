"""FilterPy compatibility layer.

Provides filterpy-compatible function signatures that delegate to tinyops.ops.
"""

from tinygrad import Tensor

from tinyops.ops.signal.discrete_white_noise_matrix import discrete_white_noise_matrix as _q_discrete
from tinyops.ops.signal.merwe_scaled_sigma_points import merwe_scaled_sigma_points as _merwe_sigma


class _Common:
    """Namespace mimicking filterpy.common."""

    @staticmethod
    def Q_discrete_white_noise(
        dim: int,
        dt: float = 1.0,
        var: float = 1.0,
        block_size: int = 1,
        order_by_dim: bool = True,
    ) -> Tensor:
        """Discrete constant white noise model."""
        return _q_discrete(
            dimension=dim,
            time_step=dt,
            noise_variance=var,
            block_size=block_size,
            order_by_dimension=order_by_dim,
        )


class MerweScaledSigmaPoints:
    """filterpy-compatible MerweScaledSigmaPoints class.

    Thin wrapper around tinyops.ops.signal.merwe_scaled_sigma_points.
    """

    def __init__(self, n: int, alpha: float, beta: float, kappa: float):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    def sigma_points(self, x: Tensor, P: Tensor) -> Tensor:
        """Generate sigma points."""
        points, _, _ = _merwe_sigma(
            state_mean=x,
            covariance=P,
            spread=self.alpha,
            prior_knowledge=self.beta,
            secondary_scaling=self.kappa,
        )
        return points

    def weights(self) -> tuple[Tensor, Tensor]:
        """Return mean and covariance weights."""
        dummy_mean = Tensor.zeros(self.n)
        dummy_cov = Tensor.eye(self.n)
        _, mean_weights, cov_weights = _merwe_sigma(
            state_mean=dummy_mean,
            covariance=dummy_cov,
            spread=self.alpha,
            prior_knowledge=self.beta,
            secondary_scaling=self.kappa,
        )
        return mean_weights, cov_weights


class _Kalman:
    """Namespace mimicking filterpy.kalman."""
    MerweScaledSigmaPoints = MerweScaledSigmaPoints


common = _Common()
kalman = _Kalman()