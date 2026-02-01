from tinygrad import Tensor, dtypes


def percentile(
    a: Tensor,
    q: float | list[float] | Tensor,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    method: str = "linear",
) -> Tensor:
    """
    Compute the q-th percentile of the data along the specified axis.

    Returns the q-th percentile(s) of the array elements.

    Args:
        a: Input tensor.
        q: Percentile or sequence of percentiles to compute, which must be between 0 and 100 inclusive.
        axis: Axis or axes along which the percentiles are computed. The default is to compute
            the percentile(s) along a flattened version of the array.
        keepdims: If this is set to True, the axes which are reduced are left in the
            result as dimensions with size one.
        method: Interpolation method to use when the desired percentile lies between two data points.
            Only "linear" is supported: ``i + (j - i) * fraction``, where ``i`` and ``j``
            are the indices surrounding the percentile.

    Returns:
        The q-th percentile(s) of the input data.
    """
    if method != "linear":
        raise NotImplementedError("Only 'linear' interpolation is supported currently")

    scalar_q = False
    if isinstance(q, (int, float)):
        q = [float(q)]
        scalar_q = True
    elif isinstance(q, list):
        q = [float(x) for x in q]

    # Ensure q is a Tensor
    if not isinstance(q, Tensor):
        q_t = Tensor(q, device=a.device, dtype=a.dtype)
    else:
        q_t = q.cast(a.dtype)
        if len(q_t.shape) == 0:
            q_t = q_t.reshape(1)
            scalar_q = True
        elif len(q_t.shape) > 1:
            raise ValueError("q must be 1D or scalar")

    if axis is None:
        a = a.flatten()
        axis = 0

    ndim = len(a.shape)
    if axis < 0:
        axis += ndim

    # Move axis to last dim
    if axis != ndim - 1:
        perm = [i for i in range(ndim) if i != axis] + [axis]
        a = a.permute(perm)

    sorted_a, _ = a.sort()

    # a is now (..., N)
    N = sorted_a.shape[-1]

    # Calculate indices
    # q is (Q,)
    indices = (N - 1) * q_t / 100.0

    lower = indices.floor()
    upper = indices.ceil()
    fraction = indices - lower

    lower_idx = lower.cast(dtype=dtypes.int32)  # (Q,)
    upper_idx = upper.cast(dtype=dtypes.int32)  # (Q,)

    # Expand sorted_a to (Q, B..., N)
    # sorted_a: (B..., N)
    # unsqueeze 0: (1, B..., N)
    # expand: (Q, B..., N)
    sorted_a_exp = sorted_a.unsqueeze(0).expand([q_t.shape[0]] + list(sorted_a.shape))

    # Expand indices to (Q, B..., 1)
    target_shape = [q_t.shape[0]] + list(sorted_a.shape[:-1]) + [1]

    lower_idx_exp = lower_idx.reshape((q_t.shape[0],) + (1,) * (ndim - 1) + (1,)).expand(target_shape)
    upper_idx_exp = upper_idx.reshape((q_t.shape[0],) + (1,) * (ndim - 1) + (1,)).expand(target_shape)

    lower_vals = sorted_a_exp.gather(-1, lower_idx_exp).squeeze(-1)
    upper_vals = sorted_a_exp.gather(-1, upper_idx_exp).squeeze(-1)

    # Fraction reshaping for broadcasting
    fraction = fraction.reshape((q_t.shape[0],) + (1,) * (ndim - 1))

    res = lower_vals + (upper_vals - lower_vals) * fraction

    if scalar_q:
        res = res.squeeze(0)
        if keepdims:
            res = res.unsqueeze(axis)
    else:
        if keepdims:
            # res is (Q, B...)
            # We need to insert 1 at axis position (shifted by 1 due to Q)
            res = res.unsqueeze(axis + 1)

    return res
