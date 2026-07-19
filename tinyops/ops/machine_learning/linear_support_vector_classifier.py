from tinygrad import Tensor

from tinyops.ops.machine_learning._linear_support_vector import _linear_support_vector_decision


def linear_support_vector_classifier(
    samples: Tensor,
    coefficients: Tensor,
    intercept: Tensor,
) -> Tensor:
    """Compute the decision function of a linear SVM classifier.

    Args:
        samples: Input samples (n_samples, n_features).
        coefficients: Hyperplane coefficients (n_classes, n_features).
        intercept: Intercept term (n_classes,).

    Returns:
        Decision function values.
    """
    return _linear_support_vector_decision(samples, coefficients, intercept)
