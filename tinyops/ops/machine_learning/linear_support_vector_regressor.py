from tinygrad import Tensor

from tinyops.ops.machine_learning._linear_support_vector import _linear_support_vector_decision


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
    return _linear_support_vector_decision(samples, coefficients, intercept).flatten()
