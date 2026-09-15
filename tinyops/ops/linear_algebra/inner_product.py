import math

from tinygrad import Tensor


def inner_product(first: Tensor, second: Tensor) -> Tensor:
    """Inner product of two arrays.

    For 1D inputs this is equivalent to the dot product. For higher
    dimensions it sums over the last axis of both inputs.

    Args:
        first: First input tensor.
        second: Second input tensor.

    Returns:
        Inner product result tensor.
    """
    if len(first.shape) == 0 or len(second.shape) == 0:
        return first * second

    if first.shape[-1] != second.shape[-1]:
        raise ValueError(
            f"shapes {first.shape} and {second.shape} not aligned: {first.shape[-1]} != {second.shape[-1]}"
        )

    contraction_size = first.shape[-1]

    product_first = math.prod(first.shape[:-1])

    product_second = math.prod(second.shape[:-1])

    flat_first = first.reshape(product_first, contraction_size)
    flat_second = second.reshape(product_second, contraction_size)

    result = flat_first.matmul(flat_second.transpose(1, 0))
    final_shape = list(first.shape[:-1]) + list(second.shape[:-1])
    return result.reshape(final_shape)
