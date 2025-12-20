from tinygrad import Tensor

def kron(a: Tensor, b: Tensor) -> Tensor:
    """
    Kronecker product of two arrays.
    """
    ndim = max(len(a.shape), len(b.shape))

    # Pad shapes with 1s at the beginning to match ndim
    # Note: reshape can handle this if we just use the calculated full shape directly?
    # But we need to insert 1s.

    # Use lists to build shapes
    a_shape = list(a.shape)
    b_shape = list(b.shape)

    a_shape = [1] * (ndim - len(a_shape)) + a_shape
    b_shape = [1] * (ndim - len(b_shape)) + b_shape

    a_expanded_shape = []
    b_expanded_shape = []
    final_shape = []

    for i in range(ndim):
        a_expanded_shape.extend([a_shape[i], 1])
        b_expanded_shape.extend([1, b_shape[i]])
        final_shape.append(a_shape[i] * b_shape[i])

    a_reshaped = a.reshape(tuple(a_expanded_shape))
    b_reshaped = b.reshape(tuple(b_expanded_shape))

    res = a_reshaped * b_reshaped
    return res.reshape(tuple(final_shape))
