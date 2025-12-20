from tinygrad import Tensor

def dot(a: Tensor, b: Tensor) -> Tensor:
    """
    Dot product of two arrays.
    """
    # Scalar multiplication
    if len(a.shape) == 0 or len(b.shape) == 0:
        return a * b

    if len(b.shape) == 1:
        # Inner product or sum product over last axis
        return (a * b).sum(axis=-1)

    # b is at least 2D
    # Sum product over last axis of a and second-to-last axis of b

    M = a.shape[-1]
    if b.shape[-2] != M:
        raise ValueError(f"shapes {a.shape} and {b.shape} not aligned: {M} (dim -1) != {b.shape[-2]} (dim -2)")

    dim_b = len(b.shape)
    # Permute b to bring the contracting dimension (dim -2) to the front
    # New order: [dim -2, 0, 1, ..., dim -3, dim -1]
    perm = [dim_b - 2] + list(range(dim_b - 2)) + [dim_b - 1]
    b_permuted = b.permute(perm)

    prod_A = 1
    for s in a.shape[:-1]: prod_A *= s

    prod_B = 1
    for s in b.shape[:-2]: prod_B *= s

    N = b.shape[-1]

    # Flatten to perform matrix multiplication
    # We use -1 for prod_A to handle potential dimension 0 issues safely if tinygrad allows?
    # Actually tinygrad reshape needs exact sizes usually or one -1.
    # Let's use the calculated prods.

    flat_a = a.reshape(prod_A, M)
    flat_b = b_permuted.reshape(M, prod_B * N)

    res = flat_a.matmul(flat_b)

    # Reshape back to expected output shape
    final_shape = list(a.shape[:-1]) + list(b.shape[:-2]) + [N]
    return res.reshape(final_shape)
