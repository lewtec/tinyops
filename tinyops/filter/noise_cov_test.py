import unittest
import pytest
import numpy as np
from tinygrad import Tensor
from tinyops.filter.noise_cov import noise_cov
from tinyops._core import assert_close
from filterpy.common import Q_discrete_white_noise

def test_noise_cov_dim2_block1():
    dt = 0.1
    var = 1.0
    dim = 2
    block_size = 1

    expected = Q_discrete_white_noise(dim=dim, dt=dt, var=var, block_size=block_size)
    result = noise_cov(dim, dt, var, block_size)

    assert_close(result, expected)

def test_noise_cov_dim3_block1():
    dt = 0.1
    var = 0.5
    dim = 3
    block_size = 1

    expected = Q_discrete_white_noise(dim=dim, dt=dt, var=var, block_size=block_size)
    result = noise_cov(dim, dt, var, block_size)

    assert_close(result, expected)

def test_noise_cov_dim2_block2_order_dim():
    dt = 0.1
    var = 1.0
    dim = 2
    block_size = 2
    order_by_dim = True

    expected = Q_discrete_white_noise(dim=dim, dt=dt, var=var, block_size=block_size, order_by_dim=order_by_dim)
    result = noise_cov(dim, dt, var, block_size, order_by_dim)

    assert_close(result, expected)

def test_noise_cov_dim2_block2_order_deriv():
    dt = 0.1
    var = 1.0
    dim = 2
    block_size = 2
    order_by_dim = False

    expected = Q_discrete_white_noise(dim=dim, dt=dt, var=var, block_size=block_size, order_by_dim=order_by_dim)
    result = noise_cov(dim, dt, var, block_size, order_by_dim)

    assert_close(result, expected)

def test_noise_cov_dim4():
    dt = 0.05
    var = 2.0
    dim = 4

    expected = Q_discrete_white_noise(dim=dim, dt=dt, var=var)
    result = noise_cov(dim, dt, var)

    assert_close(result, expected)

def test_tensor_inputs():
    dt = Tensor(0.1)
    var = Tensor(1.0)
    dim = 2

    result = noise_cov(dim, dt, var)
    expected = Q_discrete_white_noise(dim=dim, dt=0.1, var=1.0)

    assert_close(result, expected)
