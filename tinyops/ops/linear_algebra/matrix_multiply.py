from tinygrad import Tensor


def matrix_multiply(first: Tensor, second: Tensor) -> Tensor:
    """Matrix product of two arrays.

    Args:
        first: First input tensor (must not be scalar).
        second: Second input tensor (must not be scalar).

    Returns:
        Matrix product result.

    Raises:
        ValueError: If either input is a scalar.
    """
    if len(first.shape) == 0 or len(second.shape) == 0:
        raise ValueError("Scalar operands are not allowed, use '*' instead")
    return first.matmul(second)
