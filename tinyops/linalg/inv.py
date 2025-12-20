from tinygrad import Tensor
def inv(a: Tensor) -> Tensor:
    """Multiplicative inverse using Newton-Schulz."""
    if len(a.shape) < 2: raise ValueError("Must be >= 2D")
    n = a.shape[-1]
    if a.shape[-2] != n: raise ValueError("Must be square")
    perm = list(range(len(a.shape))); perm[-1], perm[-2] = perm[-2], perm[-1]
    a_t = a.permute(perm)
    a_abs = a.abs()
    norm_1 = a_abs.sum(axis=-2).max(axis=-1)
    norm_inf = a_abs.sum(axis=-1).max(axis=-1)
    new_shape = list(norm_1.shape) + [1, 1]
    norm_1 = norm_1.reshape(new_shape)
    norm_inf = norm_inf.reshape(new_shape)
    x = a_t / (norm_1 * norm_inf)
    I = Tensor.eye(n)
    for _ in range(20):
        term = (2 * I) - a.matmul(x)
        x = x.matmul(term)
    return x
