from tinygrad import Tensor

from tinyops.ops.machine_learning._svm_kernel import KernelType, _kernel_support_vector_decision


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
    return _kernel_support_vector_decision(
        samples=samples,
        support_vectors=support_vectors,
        dual_coefficients=dual_coefficients,
        intercept=intercept,
        kernel=kernel,
        polynomial_degree=polynomial_degree,
        gamma=gamma,
        coefficient_zero=coefficient_zero,
    )
