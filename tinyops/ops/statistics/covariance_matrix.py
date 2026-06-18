from tinygrad import Tensor


def covariance_matrix(
    observations: Tensor,
    second_observations: Tensor | None = None,
    rows_are_variables: bool = True,
    degrees_of_freedom: int = 1,
) -> Tensor:
    """Estimate a covariance matrix from observation data.

    Args:
        observations: Input tensor. Each row (or column when
            *rows_are_variables* is True) represents a variable and each
            column (or row) an observation.
        second_observations: Optional second set of observations to stack
            with *observations* before computing.
        rows_are_variables: If True, each row is a variable.
            If False, each column is a variable (transposed internally).
        degrees_of_freedom: The divisor is ``N - degrees_of_freedom``.

    Returns:
        Covariance matrix tensor.
    """
    matrix = observations
    if second_observations is not None:
        matrix = Tensor.stack([matrix, second_observations])

    if not rows_are_variables and matrix.shape[0] != 1:
        matrix = matrix.permute(1, 0)

    if matrix.shape[0] == 0:
        return Tensor([])

    if matrix.ndim > 2:
        raise ValueError("observations has more than 2 dimensions")

    row_means = matrix.mean(axis=1, keepdim=True)
    centered = matrix - row_means
    sample_count = matrix.shape[1]

    if degrees_of_freedom == 0:
        divisor = sample_count
    else:
        divisor = sample_count - degrees_of_freedom

    if divisor == 0:
        return Tensor.full(matrix.shape[0], matrix.shape[0], float("nan"))

    return (centered @ centered.T) / divisor
