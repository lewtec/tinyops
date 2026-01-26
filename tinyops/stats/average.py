from tinygrad import Tensor


def average(
    a: Tensor, axis: int | tuple[int, ...] | None = None, weights: Tensor | None = None, returned: bool = False
) -> Tensor | tuple[Tensor, Tensor]:
    """
    Compute the weighted average along the specified axis.
    """
    if weights is None:
        avg = a.mean(axis=axis)
        if returned:
            if axis is None:
                cnt = 1
                for s in a.shape:
                    cnt *= s
            elif isinstance(axis, int):
                cnt = a.shape[axis]
            else:
                cnt = 1
                for ax in axis:
                    cnt *= a.shape[ax]

            scl = Tensor(cnt, dtype=a.dtype, device=a.device)
            return avg, scl
        return avg

    wgt = weights
    # Normalize axis
    ndim = len(a.shape)

    # Handle broadcasting for 1D weights if axis is specified
    if axis is not None and len(wgt.shape) == 1:
        if isinstance(axis, int):
            ax = axis
            if ax < 0:
                ax += ndim

            if wgt.shape[0] != a.shape[ax]:
                raise ValueError(
                    f"Length of weights ({wgt.shape[0]}) not compatible with specified axis ({a.shape[ax]})"
                )

            # Reshape wgt to broadcast
            new_shape = [1] * ndim
            new_shape[ax] = wgt.shape[0]
            wgt = wgt.reshape(tuple(new_shape))
        else:
            # Axis is tuple, wgt is 1D.
            # Numpy behavior? Likely raises error unless broadcastable normally.
            # We assume standard broadcasting if not simple axis case.
            pass

    scl = wgt.sum(axis=axis)
    # Note: sum(a*w) / sum(w)
    # Check if scl contains 0? Tinygrad handles div by zero (inf/nan).

    prod = a * wgt
    avg = prod.sum(axis=axis) / scl

    if returned:
        return avg, scl

    return avg
