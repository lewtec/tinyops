from tinygrad import Tensor


def mse(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Mean Squared Error regression loss.
    """
    return ((y_true - y_pred) ** 2).mean()
