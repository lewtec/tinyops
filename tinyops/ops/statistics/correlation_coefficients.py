from tinygrad import Tensor

from tinyops.ops.statistics.covariance_matrix import covariance_matrix

NUMERICAL_STABILITY_EPSILON = 1e-10


def correlation_coefficients(
    observations: Tensor,
    second_observations: Tensor | None = None,
    rows_are_variables: bool = True,
) -> Tensor:
    """Compute Pearson product-moment correlation coefficients.

    Args:
        observations: Input tensor where each row (or column) is a variable.
        second_observations: Optional second variable to correlate with.
        rows_are_variables: If True, each row is treated as a variable.

    Returns:
        Correlation coefficient matrix.
    """
    covariance = covariance_matrix(observations, second_observations, rows_are_variables)
    diagonal_elements = covariance.diagonal()
    standard_deviations = diagonal_elements.sqrt()
    normalization = standard_deviations.unsqueeze(1) @ standard_deviations.unsqueeze(0)
    return covariance / (normalization + NUMERICAL_STABILITY_EPSILON)
