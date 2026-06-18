from collections.abc import Sequence

from tinygrad import Tensor


def tensor_dot_product(
    first: Tensor,
    second: Tensor,
    axes: int | Sequence[int | Sequence[int]] = 2,
) -> Tensor:
    """Compute tensor dot product along specified axes.

    Args:
        first: First input tensor.
        second: Second input tensor.
        axes: Number of axes to contract, or explicit axes specification.

    Returns:
        Tensor dot product result.
    """
    if isinstance(axes, int):
        first_axes = list(range(len(first.shape) - axes, len(first.shape)))
        second_axes = list(range(0, axes))
    else:
        raw_first, raw_second = axes
        first_axes = [raw_first] if isinstance(raw_first, int) else list(raw_first)
        second_axes = [raw_second] if isinstance(raw_second, int) else list(raw_second)

    first_dimensions = len(first.shape)
    second_dimensions = len(second.shape)
    first_axes = [ax + first_dimensions if ax < 0 else ax for ax in first_axes]
    second_axes = [ax + second_dimensions if ax < 0 else ax for ax in second_axes]

    if len(first_axes) != len(second_axes):
        raise ValueError("Different number of contraction axes")
    for i, j in zip(first_axes, second_axes):
        if first.shape[i] != second.shape[j]:
            raise ValueError(f"Shape mismatch: {first.shape[i]} != {second.shape[j]}")

    free_first = [i for i in range(first_dimensions) if i not in first_axes]
    free_second = [i for i in range(second_dimensions) if i not in second_axes]

    permuted_first = first.permute(free_first + first_axes)
    permuted_second = second.permute(second_axes + free_second)

    product_free_first = 1
    for i in free_first:
        product_free_first *= first.shape[i]
    product_contract = 1
    for i in first_axes:
        product_contract *= first.shape[i]
    product_free_second = 1
    for i in free_second:
        product_free_second *= second.shape[i]

    flat_first = permuted_first.reshape(product_free_first, product_contract)
    flat_second = permuted_second.reshape(product_contract, product_free_second)

    result = flat_first.matmul(flat_second)
    return result.reshape([first.shape[i] for i in free_first] + [second.shape[i] for i in free_second])
