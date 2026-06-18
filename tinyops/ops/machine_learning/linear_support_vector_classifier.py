from tinygrad import Tensor


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
    return samples @ coefficients.T + intercept
