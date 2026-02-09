import pytest
from filterpy.common import Q_discrete_white_noise
from tinygrad import Tensor

from tinyops._core import assert_close, assert_one_kernel
from tinyops.signal.q_discrete_white_noise import q_discrete_white_noise

TEST_PARAMS = [
    (2, 0.1, 1.0, 1, None),
    (3, 0.1, 0.5, 1, None),
    (2, 0.1, 1.0, 2, True),
    (2, 0.1, 1.0, 2, False),
    (4, 0.05, 2.0, 1, None),
]


@pytest.mark.parametrize("dim,dt,var,block_size,order_by_dim", TEST_PARAMS)
@assert_one_kernel
def test_q_discrete_white_noise(dim, dt, var, block_size, order_by_dim):
    if order_by_dim is None:
        expected = Q_discrete_white_noise(dim=dim, dt=dt, var=var, block_size=block_size)
        result = q_discrete_white_noise(dim, dt, var, block_size).realize()
    else:
        expected = Q_discrete_white_noise(dim=dim, dt=dt, var=var, block_size=block_size, order_by_dim=order_by_dim)
        result = q_discrete_white_noise(dim, dt, var, block_size, order_by_dim).realize()

    assert_close(result, expected)


@assert_one_kernel
def test_tensor_inputs():
    dt = Tensor(0.1).realize()
    var = Tensor(1.0).realize()
    dim = 2

    result = q_discrete_white_noise(dim, dt, var).realize()
    expected = Q_discrete_white_noise(dim=dim, dt=0.1, var=1.0)

    assert_close(result, expected)
