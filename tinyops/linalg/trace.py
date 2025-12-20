from tinygrad import Tensor

def trace(a: Tensor, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Tensor:
    """
    Return the sum along diagonals of the array.
    """
    # Normalize axes
    ndim = len(a.shape)
    if axis1 < 0: axis1 += ndim
    if axis2 < 0: axis2 += ndim

    if axis1 == axis2:
         raise ValueError("axis1 and axis2 cannot be the same")

    # Permute to move axis1, axis2 to 0, 1
    perm = [axis1, axis2] + [i for i in range(ndim) if i != axis1 and i != axis2]
    a_perm = a.permute(perm)

    n = a_perm.shape[0]
    m = a_perm.shape[1]

    if offset >= 0:
        if offset >= m:
             rem_shape = a_perm.shape[2:]
             return Tensor.zeros(*rem_shape)

        k = min(n, m - offset)
        idx1 = Tensor.arange(k)
        idx2 = Tensor.arange(k) + offset
    else:
        if -offset >= n:
             rem_shape = a_perm.shape[2:]
             return Tensor.zeros(*rem_shape)

        k = min(n + offset, m)
        idx1 = Tensor.arange(k) - offset
        idx2 = Tensor.arange(k)

    d = a_perm[idx1, idx2]
    return d.sum(axis=0)
