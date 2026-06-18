from tinygrad import Tensor


def vector_dot_product(first: Tensor, second: Tensor) -> Tensor:
    """Dot product of two vectors (flattened inputs).

    Args:
        first: First input tensor (flattened before computation).
        second: Second input tensor (flattened before computation).

    Returns:
        Scalar tensor containing the dot product.
    """
    return (first.flatten() * second.flatten()).sum()
