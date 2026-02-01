from tinygrad import Tensor


def median(a: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
    """
    Compute the median along the specified axis.

    The median is the value separating the higher half from the lower half of a data sample.
    This implementation uses sorting, which provides deterministic results but requires
    O(N log N) memory and computation.

    Args:
        a: Input tensor.
        axis: Axis or axes along which the median is computed. The default is to compute
            the median along a flattened version of the array.
        keepdims: If this is set to True, the axes which are reduced are left in the
            result as dimensions with size one.

    Returns:
        The median of the array elements.
    """
    if axis is None:
        a = a.flatten()
        axis = 0

    ndim = len(a.shape)
    if axis < 0:
        axis += ndim

    if axis < 0 or axis >= ndim:
        raise ValueError(f"Axis {axis} out of bounds for array of dimension {ndim}")

    # Move axis to last dimension
    if axis != ndim - 1:
        # perm: all indices except axis, then axis
        perm = [i for i in range(ndim) if i != axis] + [axis]
        a = a.permute(perm)

    # Sort along last dimension
    # sort() returns (values, indices)
    sorted_a, _ = a.sort()

    k = sorted_a.shape[-1]

    if k % 2 == 1:
        mid = (k - 1) // 2
        res = sorted_a[..., mid : mid + 1]
    else:
        mid1 = k // 2 - 1
        mid2 = k // 2
        v1 = sorted_a[..., mid1 : mid1 + 1]
        v2 = sorted_a[..., mid2 : mid2 + 1]
        res = (v1 + v2) / 2

    if keepdims:
        if axis != ndim - 1:
            # Move last dimension back to axis
            inv_perm = list(range(axis)) + [ndim - 1] + list(range(axis, ndim - 1))
            res = res.permute(inv_perm)
    else:
        res = res.squeeze(-1)

    return res
