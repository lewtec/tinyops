from tinygrad import Tensor

from tinyops.ops.linear_algebra.inverse import inverse


def matrix_power(matrix: Tensor, exponent: int) -> Tensor:
    """Raise a square matrix to an integer power.

    Uses binary exponentiation for efficiency. Negative exponents
    compute the inverse first.

    Args:
        matrix: A square matrix tensor (at least 2D).
        exponent: Integer power.

    Returns:
        The matrix raised to the given power.
    """
    if len(matrix.shape) < 2:
        raise ValueError("matrix must have at least 2 dimensions")
    if matrix.shape[-2] != matrix.shape[-1]:
        raise ValueError("Last 2 dimensions of the array must be square")

    size = matrix.shape[-1]

    if exponent == 0:
        identity = Tensor.eye(size, dtype=matrix.dtype, device=matrix.device)
        batch_shape = matrix.shape[:-2]
        if batch_shape:
            identity = identity.reshape((1,) * len(batch_shape) + (size, size))
            identity = identity.expand(matrix.shape)
        return identity

    if exponent < 0:
        matrix = inverse(matrix)
        exponent = abs(exponent)

    result = None
    current = matrix
    while exponent > 0:
        if exponent % 2 == 1:
            result = current if result is None else result.matmul(current)
        exponent //= 2
        if exponent > 0:
            current = current.matmul(current)

    return result
