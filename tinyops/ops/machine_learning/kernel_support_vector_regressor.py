from tinygrad import Tensor

from tinyops.ops.machine_learning.kernel_support_vector_classifier import KernelType


def kernel_support_vector_regressor(
    samples: Tensor,
    support_vectors: Tensor,
    dual_coefficients: Tensor,
    intercept: Tensor,
    kernel: KernelType = KernelType.RADIAL_BASIS_FUNCTION,
    polynomial_degree: int = 3,
    gamma: float | str = "scale",
    coefficient_zero: float = 0.0,
) -> Tensor:
    """Compute the prediction of a kernel SVR model.

    Args:
        samples: Input samples (n_samples, n_features).
        support_vectors: Support vectors from training.
        dual_coefficients: Dual coefficients for support vectors.
        intercept: Intercept term.
        kernel: Kernel function type.
        polynomial_degree: Degree for polynomial kernel.
        gamma: Kernel coefficient ('scale', 'auto', or float).
        coefficient_zero: Independent term in kernel function.

    Returns:
        Predicted values.
    """
    if gamma == "scale":
        gamma = 1.0 / (samples.shape[1] * samples.var()) if samples.shape[1] > 0 else 1.0
    elif gamma == "auto":
        gamma = 1.0 / samples.shape[1] if samples.shape[1] > 0 else 1.0

    if kernel == KernelType.LINEAR:
        kernel_matrix = samples @ support_vectors.T
    elif kernel == KernelType.POLYNOMIAL:
        kernel_matrix = ((samples @ support_vectors.T) * gamma + coefficient_zero).pow(polynomial_degree)
    elif kernel == KernelType.RADIAL_BASIS_FUNCTION:
        kernel_matrix = (-gamma * (samples.unsqueeze(1) - support_vectors.unsqueeze(0)).pow(2).sum(-1)).exp()
    elif kernel == KernelType.SIGMOID:
        kernel_matrix = ((samples @ support_vectors.T) * gamma + coefficient_zero).tanh()
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    return kernel_matrix @ dual_coefficients.T + intercept
