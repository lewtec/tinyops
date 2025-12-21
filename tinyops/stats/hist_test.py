import numpy as np
from tinygrad import Tensor
from tinyops.stats.hist import hist
from tinyops._core import assert_close
import pytest

def test_hist_basic():
    x_np = np.random.randn(100).astype(np.float32)
    x = Tensor(x_np)

    h, e = hist(x, bins=10)
    h_np, e_np = np.histogram(x_np, bins=10)

    assert_close(h, h_np)
    assert_close(e, e_np)

def test_hist_range():
    x_np = np.array([0, 1, 2, 3, 4, 10], dtype=np.float32)
    x = Tensor(x_np)

    # Range excluding 10
    h, e = hist(x, bins=5, range=(0, 5))
    h_np, e_np = np.histogram(x_np, bins=5, range=(0, 5))

    assert_close(h, h_np)
    assert_close(e, e_np)

def test_hist_density():
    x_np = np.random.randn(100).astype(np.float32)
    x = Tensor(x_np)

    h, e = hist(x, bins=10, density=True)
    h_np, e_np = np.histogram(x_np, bins=10, density=True)

    assert_close(h, h_np)
    assert_close(e, e_np)

def test_hist_empty():
    x_np = np.array([], dtype=np.float32)
    x = Tensor(x_np)

    h, e = hist(x, bins=5)
    h_np, e_np = np.histogram(x_np, bins=5)

    assert_close(h, h_np)
    assert_close(e, e_np)

def test_hist_edge_values():
    # Values exactly on boundaries
    x_np = np.array([0, 1, 2, 3], dtype=np.float32)
    x = Tensor(x_np)
    # range 0-3, bins 3 -> [0,1), [1,2), [2,3]
    # 3 should be in last bin [2,3]

    h, e = hist(x, bins=3, range=(0, 3))
    h_np, e_np = np.histogram(x_np, bins=3, range=(0, 3))

    assert_close(h, h_np)
    assert_close(e, e_np)

def test_hist_flat_region():
    x_np = np.ones(10, dtype=np.float32)
    x = Tensor(x_np)

    h, e = hist(x, bins=5)
    h_np, e_np = np.histogram(x_np, bins=5)

    assert_close(h, h_np)
    assert_close(e, e_np)
