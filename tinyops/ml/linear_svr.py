from tinygrad import Tensor

def linear_svr(x: Tensor, coef: Tensor, intercept: Tensor) -> Tensor:
  """
  Computes the prediction of a linear support vector regression.

  This function is stateless and designed to replicate the predict
  method of a trained sklearn.svm.LinearSVR model.

  Args:
    x: Input samples. Shape: (n_samples, n_features).
    coef: Coefficients of the hyperplane. Shape: (1, n_features).
    intercept: Intercept or bias term. Shape: (1,).

  Returns:
    The predicted value for each sample. Shape: (n_samples,).
  """
  return (x @ coef.T + intercept).flatten()
