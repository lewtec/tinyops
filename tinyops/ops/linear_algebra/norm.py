from tinygrad import Tensor


def _vector_norm(
    tensor: Tensor,
    order: int | float | str | None,
    axis: int,
    keep_dimensions: bool,
) -> Tensor:
    if order is None:
        order = 2
    if order == float("inf"):
        return tensor.abs().max(axis=axis, keepdim=keep_dimensions)
    elif order == float("-inf"):
        return tensor.abs().min(axis=axis, keepdim=keep_dimensions)
    elif order == 0:
        return (tensor != 0).sum(axis=axis, keepdim=keep_dimensions)
    else:
        return (tensor.abs() ** order).sum(axis=axis, keepdim=keep_dimensions) ** (1 / order)


def _matrix_norm(
    tensor: Tensor,
    order: int | float | str | None,
    axis: tuple[int, int],
    keep_dimensions: bool,
    number_of_dimensions: int,
) -> Tensor:
    row_axis, column_axis = axis
    if row_axis < 0:
        row_axis += number_of_dimensions
    if column_axis < 0:
        column_axis += number_of_dimensions

    if order == "fro" or order is None:
        return (tensor.abs() ** 2).sum(axis=axis, keepdim=keep_dimensions).sqrt()

    if order == float("inf"):
        intermediate = tensor.abs().sum(axis=column_axis, keepdim=True)
        result = intermediate.max(axis=row_axis, keepdim=True)
    elif order == float("-inf"):
        intermediate = tensor.abs().sum(axis=column_axis, keepdim=True)
        result = intermediate.min(axis=row_axis, keepdim=True)
    elif order == 1:
        intermediate = tensor.abs().sum(axis=row_axis, keepdim=True)
        result = intermediate.max(axis=column_axis, keepdim=True)
    elif order == -1:
        intermediate = tensor.abs().sum(axis=row_axis, keepdim=True)
        result = intermediate.min(axis=column_axis, keepdim=True)
    elif order == 2:
        raise NotImplementedError("Spectral norm (order=2) not supported.")
    else:
        raise ValueError(f"Invalid norm order for matrices: {order}")

    if not keep_dimensions:
        axes = sorted((row_axis, column_axis), reverse=True)
        for ax in axes:
            result = result.squeeze(ax)

    return result


def norm(
    tensor: Tensor,
    order: int | float | str | None = None,
    axis: int | tuple[int, ...] | None = None,
    keep_dimensions: bool = False,
) -> Tensor:
    """Compute matrix or vector norm.

    Args:
        tensor: Input tensor.
        order: Order of the norm. Supports numeric orders, ``'fro'``,
            ``inf`` and ``-inf``.
        axis: Axis or axes along which to compute. Determines vector vs
            matrix norm semantics.
        keep_dimensions: If True, reduced axes are kept as size-one dimensions.

    Returns:
        Norm value tensor.
    """
    number_of_dimensions = len(tensor.shape)

    if axis is None:
        if number_of_dimensions == 1:
            if order is None:
                order = 2
            axis = 0
        elif number_of_dimensions == 2:
            if order is None:
                order = "fro"
            axis = (0, 1)
        else:
            if order is None:
                tensor = tensor.flatten()
                number_of_dimensions = 1
                axis = 0
                order = 2
            else:
                raise ValueError("Improper number of dimensions to norm.")

    # Vector norms
    if isinstance(axis, int):
        return _vector_norm(tensor, order, axis, keep_dimensions)

    # Matrix norms
    if isinstance(axis, (tuple, list)) and len(axis) == 2:
        return _matrix_norm(tensor, order, axis, keep_dimensions, number_of_dimensions)

    raise ValueError("Invalid axis argument")
