from tinygrad import Tensor


def linear_support_vector_regressor(
    samples: Tensor,
    coefficients: Tensor,
    intercept: Tensor,
) -> Tensor:
    """Compute the prediction of a linear SVR model.

    Args:
        samples: Input samples (n_samples, n_features).
        coefficients: Hyperplane coefficients (1, n_features).
        intercept: Intercept term (1,).

    Returns:
        Predicted values (n_samples,).
    """
    return (samples @ coefficients.T + intercept).flatten()
