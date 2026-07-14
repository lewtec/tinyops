from enum import Enum

from tinygrad import Tensor


class KernelType(Enum):
    """Kernel functions for SVM."""

    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    RADIAL_BASIS_FUNCTION = "radial_basis_function"
    SIGMOID = "sigmoid"


def _compute_kernel_matrix(
    samples: Tensor,
    support_vectors: Tensor,
    kernel: KernelType,
    polynomial_degree: int,
    gamma: float | str,
    coefficient_zero: float,
) -> Tensor:
    """Compute the kernel matrix for SVM models."""
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

    return kernel_matrix



def _kernel_support_vector_decision(
    samples: Tensor,
    support_vectors: Tensor,
    dual_coefficients: Tensor,
    intercept: Tensor,
    kernel: KernelType = KernelType.RADIAL_BASIS_FUNCTION,
    polynomial_degree: int = 3,
    gamma: float | str = "scale",
    coefficient_zero: float = 0.0,
) -> Tensor:
    """Shared decision path for kernel SVM classifier and regressor."""
    kernel_matrix = _compute_kernel_matrix(
        samples=samples,
        support_vectors=support_vectors,
        kernel=kernel,
        polynomial_degree=polynomial_degree,
        gamma=gamma,
        coefficient_zero=coefficient_zero,
    )
    return kernel_matrix @ dual_coefficients.T + intercept
