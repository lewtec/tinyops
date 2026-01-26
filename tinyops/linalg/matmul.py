from tinygrad import Tensor


def matmul(x1: Tensor, x2: Tensor) -> Tensor:
    """
    Matrix product of two arrays.
    """
    if len(x1.shape) == 0 or len(x2.shape) == 0:
        raise ValueError("Scalar operands are not allowed, use '*' instead")

    return x1.matmul(x2)
