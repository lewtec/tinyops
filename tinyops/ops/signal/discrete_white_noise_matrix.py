from tinygrad import Tensor

from tinyops.ops.linear_algebra.kronecker_product import kronecker_product


def _build_base_matrix(dimension: int, dt: Tensor) -> Tensor:
    """Construct the base block for the discrete white noise matrix."""
    one = Tensor(1.0, dtype=dt.dtype, device=dt.device)
    dt2, dt3, dt4 = dt**2, dt**3, dt**4

    if dimension == 2:
        return Tensor.stack(
            [
                Tensor.stack([dt4 / 4, dt3 / 2]),
                Tensor.stack([dt3 / 2, dt2]),
            ]
        )
    if dimension == 3:
        return Tensor.stack(
            [
                Tensor.stack([dt4 / 4, dt3 / 2, dt2 / 2]),
                Tensor.stack([dt3 / 2, dt2, dt]),
                Tensor.stack([dt2 / 2, dt, one]),
            ]
        )

    dt5, dt6 = dt**5, dt**6
    return Tensor.stack(
        [
            Tensor.stack([dt6 / 36, dt5 / 12, dt4 / 6, dt3 / 6]),
            Tensor.stack([dt5 / 12, dt4 / 4, dt3 / 2, dt2 / 2]),
            Tensor.stack([dt4 / 6, dt3 / 2, dt2, dt]),
            Tensor.stack([dt3 / 6, dt2 / 2, dt, one]),
        ]
    )


def discrete_white_noise_matrix(
    dimension: int,
    time_step: float | Tensor = 1.0,
    noise_variance: float | Tensor = 1.0,
    block_size: int = 1,
    order_by_dimension: bool = True,
) -> Tensor:
    """Construct the process noise covariance matrix for a discrete constant white noise model.

    Args:
        dimension: Model dimension (2, 3, or 4).
        time_step: Time step duration.
        noise_variance: Noise variance.
        block_size: If > 1, creates a block diagonal matrix.
        order_by_dimension: If True, blocks are ordered by dimension.

    Returns:
        Process noise covariance matrix.

    Raises:
        ValueError: If dimension is not 2, 3, or 4.
    """
    if dimension not in (2, 3, 4):
        raise ValueError("dimension must be 2, 3, or 4")

    time_step_tensor = time_step if isinstance(time_step, Tensor) else Tensor(time_step)
    variance_tensor = noise_variance if isinstance(noise_variance, Tensor) else Tensor(noise_variance)

    noise_matrix = _build_base_matrix(dimension, time_step_tensor)

    if block_size == 1:
        return noise_matrix * variance_tensor

    identity = Tensor.eye(block_size, dtype=noise_matrix.dtype, device=noise_matrix.device)

    if order_by_dimension:
        result = kronecker_product(identity, noise_matrix)
    else:
        result = kronecker_product(noise_matrix, identity)

    return result * variance_tensor
