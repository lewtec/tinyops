from tinygrad import Tensor
def pinv(a: Tensor) -> Tensor:
    """Pseudo-inverse using Newton-Schulz."""
    if len(a.shape) < 2: raise ValueError("Must be >= 2D")
    m, n = a.shape[-2], a.shape[-1]
    perm = list(range(len(a.shape))); perm[-1], perm[-2] = perm[-2], perm[-1]
    a_t = a.permute(perm)
    a_abs = a.abs()
    norm_1 = a_abs.sum(axis=-2).max(axis=-1)
    norm_inf = a_abs.sum(axis=-1).max(axis=-1)
    new_shape = list(norm_1.shape) + [1, 1]
    norm_1 = norm_1.reshape(new_shape)
    norm_inf = norm_inf.reshape(new_shape)
    denom = norm_1 * norm_inf
    x = a_t / (denom + 1e-12)
    steps = 20
    if m >= n:
        I = Tensor.eye(n)
        for _ in range(steps):
             xa = x.matmul(a)
             term = (2 * I) - xa
             x = term.matmul(x)
    else:
        I = Tensor.eye(m)
        for _ in range(steps):
             ax = a.matmul(x)
             term = (2 * I) - ax
             x = x.matmul(term)
    return x
