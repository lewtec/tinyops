from typing import Literal
from tinygrad import Tensor

def svr(
    X: Tensor,
    support_vectors: Tensor,
    dual_coef: Tensor,
    intercept: Tensor,
    kernel: Literal["linear", "poly", "rbf", "sigmoid"] = "rbf",
    degree: int = 3,
    gamma: float = "scale",
    coef0: float = 0.0,
) -> Tensor:
    """
    Computes the decision function of a kernel Support Vector Regression.
    This is a stateless function that replicates the predict method
    of a trained sklearn.svm.SVR model.
    """
    if gamma == "scale":
        gamma = 1.0 / (X.shape[1] * X.var()) if X.shape[1] > 0 else 1.0
    elif gamma == "auto":
        gamma = 1.0 / X.shape[1] if X.shape[1] > 0 else 1.0

    if kernel == "linear":
        K = X @ support_vectors.T
    elif kernel == "poly":
        K = ((X @ support_vectors.T) * gamma + coef0).pow(degree)
    elif kernel == "rbf":
        K = (-gamma * (X.unsqueeze(1) - support_vectors.unsqueeze(0)).pow(2).sum(-1)).exp()
    elif kernel == "sigmoid":
        K = ((X @ support_vectors.T) * gamma + coef0).tanh()

    return K @ dual_coef.T + intercept
