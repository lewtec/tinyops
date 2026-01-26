from tinygrad import Tensor


def inner(a: Tensor, b: Tensor) -> Tensor:
    """
    Inner product of two arrays.
    """
    if len(a.shape) == 0 or len(b.shape) == 0:
        return a * b

    if a.shape[-1] != b.shape[-1]:
        raise ValueError(f"shapes {a.shape} and {b.shape} not aligned: {a.shape[-1]} != {b.shape[-1]}")

    K = a.shape[-1]

    prod_A = 1
    for s in a.shape[:-1]:
        prod_A *= s

    prod_B = 1
    for s in b.shape[:-1]:
        prod_B *= s

    flat_a = a.reshape(prod_A, K)
    flat_b = b.reshape(prod_B, K)

    # flat_b is (Prod_B, K). We want (K, Prod_B).
    res = flat_a.matmul(flat_b.transpose(1, 0))  # (Prod_A, Prod_B)

    final_shape = list(a.shape[:-1]) + list(b.shape[:-1])
    return res.reshape(final_shape)
