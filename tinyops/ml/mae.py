from tinygrad import Tensor


def mae(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Mean Absolute Error regression loss.
    """
    return (y_true - y_pred).abs().mean()
