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

    if length <= 0:
         # Handle empty result if possible, or let slice handle it
         # If length <= 0, slice start:start:... should be empty
         # Ensure end is not smaller than start if we rely on loop logic, but slice handles it.
         end = start
    else:
         step = M + 1
         end = start + length * step

    # Flatten last two dimensions
    # Note: reshape requires tuple
    new_shape = list(a_perm.shape[:-2]) + [N * M]
    a_flat = a_perm.reshape(tuple(new_shape))

    step = M + 1

    # Slice
    # We use python slicing syntax.
    # We need to construct the slice tuple to handle arbitrary dimensions + ellipsis
    # But Tensor.__getitem__ supports ellipsis.

    return a_flat[..., start:end:step]
