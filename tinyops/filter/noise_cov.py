from tinygrad import Tensor
from tinyops.linalg import kron

def noise_cov(dim: int, dt: float | Tensor = 1.0, var: float | Tensor = 1.0, block_size: int = 1, order_by_dim: bool = True) -> Tensor:
    """
    Returns the Q matrix for the Discrete Constant White Noise Model.

    Args:
        dim: Dimension for Q (2, 3, or 4).
        dt: Time step.
        var: Variance in the noise.
        block_size: If > 1, creates a block diagonal matrix.
        order_by_dim: If True, blocks are ordered by dimension. If False, by derivative.

    Returns:
        Tensor: The Q matrix.
    """
    if dim not in [2, 3, 4]:
        raise ValueError("dim must be between 2 and 4")

    dt_t = dt if isinstance(dt, Tensor) else Tensor(dt)
    var_t = var if isinstance(var, Tensor) else Tensor(var)

    # Helper for 1.0
    one = Tensor(1.0, dtype=dt_t.dtype, device=dt_t.device)

    d2 = dt_t**2
    d3 = dt_t**3
    d4 = dt_t**4
    d5 = dt_t**5
    d6 = dt_t**6

    if dim == 2:
        row0 = Tensor.stack([d4/4, d3/2])
        row1 = Tensor.stack([d3/2, d2])
        Q_t = Tensor.stack([row0, row1])
    elif dim == 3:
        row0 = Tensor.stack([d4/4, d3/2, d2/2])
        row1 = Tensor.stack([d3/2, d2, dt_t])
        row2 = Tensor.stack([d2/2, dt_t, one])
        Q_t = Tensor.stack([row0, row1, row2])
    else: # dim=4
        row0 = Tensor.stack([d6/36, d5/12, d4/6, d3/6])
        row1 = Tensor.stack([d5/12, d4/4, d3/2, d2/2])
        row2 = Tensor.stack([d4/6, d3/2, d2, dt_t])
        row3 = Tensor.stack([d3/6, d2/2, dt_t, one])
        Q_t = Tensor.stack([row0, row1, row2, row3])

    if block_size == 1:
        return Q_t * var_t

    I = Tensor.eye(block_size, dtype=Q_t.dtype, device=Q_t.device)

    if order_by_dim:
        res = kron(I, Q_t)
    else:
        res = kron(Q_t, I)

    return res * var_t
