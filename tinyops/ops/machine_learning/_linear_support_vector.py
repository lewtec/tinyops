"""Shared helpers for linear support-vector decision functions."""

from tinygrad import Tensor


def _linear_support_vector_decision(
    samples: Tensor,
    coefficients: Tensor,
    intercept: Tensor,
) -> Tensor:
    """Compute the linear SVM decision surface for samples.

    Args:
        samples: Input samples (n_samples, n_features).
        coefficients: Hyperplane coefficients (n_outputs, n_features).
        intercept: Intercept term (n_outputs,).

    Returns:
        Decision values (n_samples, n_outputs).
    """
    return samples @ coefficients.T + intercept
