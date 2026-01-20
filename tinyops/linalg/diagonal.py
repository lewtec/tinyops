from tinygrad import Tensor

def diagonal(a: Tensor, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Tensor:
    """
    Return specified diagonals.
    """
    ndim = len(a.shape)

    # Normalize axes
    if axis1 < 0: axis1 += ndim
    if axis2 < 0: axis2 += ndim

    if axis1 == axis2:
        raise ValueError("axis1 and axis2 cannot be the same")

    # Build permutation
    perm = [i for i in range(ndim) if i != axis1 and i != axis2]
    perm.append(axis1)
    perm.append(axis2)

    a_perm = a.permute(perm)

    N = a_perm.shape[-2]
    M = a_perm.shape[-1]

    if offset >= 0:
        start = offset
        length = min(N, M - offset)
    else:
        start = -offset * M
        length = min(N + offset, M)

    step = M + 1
    length = max(0, length)
    end = start + length * step

    # Flatten last two dimensions
    new_shape = a_perm.shape[:-2] + (N * M,)
    a_flat = a_perm.reshape(new_shape)

    # Slice with stride M + 1 to pick diagonal elements
    return a_flat[..., start:end:step]
