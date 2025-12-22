from tinygrad import Tensor

def linear_svc(x: Tensor, coef: Tensor, intercept: Tensor) -> Tensor:
  """
  Computes the decision function of a linear support vector classification.

  This function is stateless and designed to replicate the decision_function
  method of a trained sklearn.svm.LinearSVC model.

  Args:
    x: Input samples. Shape: (n_samples, n_features).
    coef: Coefficients of the hyperplane. Shape: (n_classes, n_features).
          For binary classification, this can be (1, n_features).
    intercept: Intercept or bias term. Shape: (n_classes,).

  Returns:
    The decision function value for each sample. Shape: (n_samples, n_classes)
    or (n_samples,) for binary classification.
  """
  return x @ coef.T + intercept
