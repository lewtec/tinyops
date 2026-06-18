from tinygrad import Tensor

from tinyops.ops.linear_algebra.kronecker_product import kronecker_product


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

    one = Tensor(1.0, dtype=time_step_tensor.dtype, device=time_step_tensor.device)

    dt2 = time_step_tensor**2
    dt3 = time_step_tensor**3
    dt4 = time_step_tensor**4
    dt5 = time_step_tensor**5
    dt6 = time_step_tensor**6

    if dimension == 2:
        noise_matrix = Tensor.stack(
            [
                Tensor.stack([dt4 / 4, dt3 / 2]),
                Tensor.stack([dt3 / 2, dt2]),
            ]
        )
    elif dimension == 3:
        noise_matrix = Tensor.stack(
            [
                Tensor.stack([dt4 / 4, dt3 / 2, dt2 / 2]),
                Tensor.stack([dt3 / 2, dt2, time_step_tensor]),
                Tensor.stack([dt2 / 2, time_step_tensor, one]),
            ]
        )
    else:
        noise_matrix = Tensor.stack(
            [
                Tensor.stack([dt6 / 36, dt5 / 12, dt4 / 6, dt3 / 6]),
                Tensor.stack([dt5 / 12, dt4 / 4, dt3 / 2, dt2 / 2]),
                Tensor.stack([dt4 / 6, dt3 / 2, dt2, time_step_tensor]),
                Tensor.stack([dt3 / 6, dt2 / 2, time_step_tensor, one]),
            ]
        )

    if block_size == 1:
        return noise_matrix * variance_tensor

    identity = Tensor.eye(block_size, dtype=noise_matrix.dtype, device=noise_matrix.device)

    if order_by_dimension:
        result = kronecker_product(identity, noise_matrix)
    else:
        result = kronecker_product(noise_matrix, identity)

    return result * variance_tensor
