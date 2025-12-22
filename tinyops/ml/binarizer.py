from tinygrad import Tensor

def binarizer(X: Tensor, threshold: float = 0.0) -> Tensor:
    return Tensor.where(X > threshold, 1.0, 0.0)
