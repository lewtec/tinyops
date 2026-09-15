from tinygrad import Tensor


def outer_product(first: Tensor, second: Tensor) -> Tensor:
    """Compute the outer product of two vectors.

    Both inputs are flattened before computation.

    Args:
        first: First input tensor.
        second: Second input tensor.

    Returns:
        Outer product matrix.
    """
    first_flat = first.flatten()
    second_flat = second.flatten()
    return first_flat.unsqueeze(1) * second_flat.unsqueeze(0)
