from tinygrad import Tensor


def kronecker_product(first: Tensor, second: Tensor) -> Tensor:
    """Compute the Kronecker product of two arrays.

    Args:
        first: First input tensor.
        second: Second input tensor.

    Returns:
        Kronecker product tensor.
    """
    max_dimensions = max(len(first.shape), len(second.shape))

    first_shape = [1] * (max_dimensions - len(first.shape)) + list(first.shape)
    second_shape = [1] * (max_dimensions - len(second.shape)) + list(second.shape)

    first_expanded_shape = []
    second_expanded_shape = []
    final_shape = []

    for i in range(max_dimensions):
        first_expanded_shape.extend([first_shape[i], 1])
        second_expanded_shape.extend([1, second_shape[i]])
        final_shape.append(first_shape[i] * second_shape[i])

    first_reshaped = first.reshape(tuple(first_expanded_shape))
    second_reshaped = second.reshape(tuple(second_expanded_shape))

    return (first_reshaped * second_reshaped).reshape(tuple(final_shape))
