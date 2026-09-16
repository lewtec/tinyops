import math

from tinygrad import Tensor


def dot_product(first: Tensor, second: Tensor) -> Tensor:
    """Compute the dot product of two arrays.

    For 1D inputs this is the inner product. For 2D+ inputs,
    it is a sum product over the last axis of *first* and the
    second-to-last axis of *second*.

    Args:
        first: First input tensor.
        second: Second input tensor.

    Returns:
        Dot product result tensor.
    """
    if len(first.shape) == 0 or len(second.shape) == 0:
        return first * second

    if len(second.shape) == 1:
        return (first * second).sum(axis=-1)

    contraction_size = first.shape[-1]
    if second.shape[-2] != contraction_size:
        raise ValueError(
            f"shapes {first.shape} and {second.shape} not aligned: "
            f"{contraction_size} (dim -1) != {second.shape[-2]} (dim -2)"
        )

    second_dimensions = len(second.shape)
    permutation = [second_dimensions - 2] + list(range(second_dimensions - 2)) + [second_dimensions - 1]
    second_permuted = second.permute(permutation)

    product_first = math.prod(first.shape[:-1])

    product_second = math.prod(second.shape[:-2])

    output_columns = second.shape[-1]

    flat_first = first.reshape(product_first, contraction_size)
    flat_second = second_permuted.reshape(contraction_size, product_second * output_columns)

    result = flat_first.matmul(flat_second)
    final_shape = list(first.shape[:-1]) + list(second.shape[:-2]) + [output_columns]
    return result.reshape(final_shape)
