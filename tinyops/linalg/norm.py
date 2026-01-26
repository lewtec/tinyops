from tinygrad import Tensor


def norm(
    x: Tensor, ord: int | float | str | None = None, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> Tensor:
    """
    Matrix or vector norm.
    """
    ndim = len(x.shape)
    # Mapping keepdims -> keepdim for tinygrad
    kd = keepdims

    if axis is None:
        if ndim == 1:
            if ord is None:
                ord = 2
            axis = 0
        elif ndim == 2:
            if ord is None:
                ord = "fro"
            axis = (0, 1)
        else:
            if ord is None:
                x = x.flatten()
                ndim = 1
                axis = 0
                ord = 2
            else:
                raise ValueError("Improper number of dimensions to norm.")

    # Handle Vector Norms
    if isinstance(axis, int):
        if ord is None:
            ord = 2

        if ord == float("inf"):
            return x.abs().max(axis=axis, keepdim=kd)
        elif ord == float("-inf"):
            return x.abs().min(axis=axis, keepdim=kd)
        elif ord == 0:
            return (x != 0).sum(axis=axis, keepdim=kd)
        else:
            return (x.abs() ** ord).sum(axis=axis, keepdim=kd) ** (1 / ord)

    # Handle Matrix Norms
    if isinstance(axis, (tuple, list)) and len(axis) == 2:
        row_axis, col_axis = axis
        # Normalize axes
        if row_axis < 0:
            row_axis += ndim
        if col_axis < 0:
            col_axis += ndim

        if ord == "fro" or ord is None:
            return (x.abs() ** 2).sum(axis=axis, keepdim=kd).sqrt()

        # For intermediate reductions, use keepdim=True

        if ord == float("inf"):
            s = x.abs().sum(axis=col_axis, keepdim=True)
            res = s.max(axis=row_axis, keepdim=True)
        elif ord == float("-inf"):
            s = x.abs().sum(axis=col_axis, keepdim=True)
            res = s.min(axis=row_axis, keepdim=True)
        elif ord == 1:
            s = x.abs().sum(axis=row_axis, keepdim=True)
            res = s.max(axis=col_axis, keepdim=True)
        elif ord == -1:
            s = x.abs().sum(axis=row_axis, keepdim=True)
            res = s.min(axis=col_axis, keepdim=True)
        elif ord == 2:
            raise NotImplementedError("Spectral norm (ord=2) not supported.")
        else:
            raise ValueError(f"Invalid norm order for matrices: {ord}")

        if not kd:
            # Squeeze axes in reverse order
            axes = sorted((row_axis, col_axis), reverse=True)
            for ax in axes:
                res = res.squeeze(ax)

        return res

    raise ValueError("Invalid axis argument")
