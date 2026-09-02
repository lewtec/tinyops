from tinygrad import Tensor

NUMERICAL_STABILITY_EPSILON = 1e-7


def non_negative_matrix_factorization(
    data: Tensor,
    component_count: int,
    maximum_iterations: int = 200,
    convergence_tolerance: float = 1e-4,
) -> tuple[Tensor, Tensor]:
    """Decompose a non-negative matrix into two non-negative factors.

    Finds W, H such that ``data ~ W @ H`` using multiplicative updates.

    Args:
        data: Input tensor (n_samples, n_features), all values non-negative.
        component_count: Number of components.
        maximum_iterations: Maximum update iterations.
        convergence_tolerance: Convergence threshold on reconstruction error.

    Returns:
        Tuple (basis_matrix, coefficient_matrix) where:
            - basis_matrix: (n_samples, component_count)
            - coefficient_matrix: (component_count, n_features)
    """
    sample_count, feature_count = data.shape

    basis = Tensor.rand(sample_count, component_count)
    coefficients = Tensor.rand(component_count, feature_count)

    previous_error = float("inf")

    for _ in range(maximum_iterations):
        # Update coefficients
        numerator_h = basis.transpose() @ data
        denominator_h = basis.transpose() @ basis @ coefficients + NUMERICAL_STABILITY_EPSILON
        coefficients = coefficients * numerator_h / denominator_h

        # Update basis
        numerator_w = data @ coefficients.transpose()
        denominator_w = basis @ coefficients @ coefficients.transpose() + NUMERICAL_STABILITY_EPSILON
        basis = basis * numerator_w / denominator_w

        error = (data - basis @ coefficients).pow(2).sum().sqrt().item()
        if abs(previous_error - error) < convergence_tolerance:
            break
        previous_error = error

    return basis, coefficients
