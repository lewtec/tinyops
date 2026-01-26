import pytest
from filterpy.common import Q_discrete_white_noise
from tinygrad import Tensor

from tinyops._core import assert_close, assert_one_kernel
from tinyops.filter.noise_cov import noise_cov

TEST_PARAMS = [
    (2, 0.1, 1.0, 1, None),
    (3, 0.1, 0.5, 1, None),
    (2, 0.1, 1.0, 2, True),
    (2, 0.1, 1.0, 2, False),
    (4, 0.05, 2.0, 1, None),
]


@pytest.mark.parametrize("dim,dt,var,block_size,order_by_dim", TEST_PARAMS)
@assert_one_kernel
def test_noise_cov(dim, dt, var, block_size, order_by_dim):
    if order_by_dim is None:
        expected = Q_discrete_white_noise(dim=dim, dt=dt, var=var, block_size=block_size)
        result = noise_cov(dim, dt, var, block_size).realize()
    else:
        expected = Q_discrete_white_noise(dim=dim, dt=dt, var=var, block_size=block_size, order_by_dim=order_by_dim)
        result = noise_cov(dim, dt, var, block_size, order_by_dim).realize()

    assert_close(result, expected)


@assert_one_kernel
def test_tensor_inputs():
    dt = Tensor(0.1).realize()
    var = Tensor(1.0).realize()
    dim = 2

    result = noise_cov(dim, dt, var).realize()
    expected = Q_discrete_white_noise(dim=dim, dt=0.1, var=1.0)

    assert_close(result, expected)
