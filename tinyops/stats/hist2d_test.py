import numpy as np
from tinygrad import Tensor

from tinyops._core import assert_close, assert_one_kernel
from tinyops.stats.hist2d import hist2d


@assert_one_kernel
def test_simple_hist2d():
    x_np = np.array([0, 1, 2, 3, 4, 5])
    y_np = np.array([0, 1, 2, 3, 4, 5])
    x_tg = Tensor(x_np).realize()
    y_tg = Tensor(y_np).realize()

    h_np, _, _ = np.histogram2d(x_np, y_np, bins=3)
    h_tg, _, _ = hist2d(x_tg, y_tg, bins=3)
    h_tg = h_tg.realize()

    assert_close(h_tg, Tensor(h_np))


@assert_one_kernel
def test_random_hist2d():
    x_np = np.random.rand(100)
    y_np = np.random.rand(100)
    x_tg = Tensor(x_np).realize()
    y_tg = Tensor(y_np).realize()

    h_np, _, _ = np.histogram2d(x_np, y_np, bins=5)
    h_tg, _, _ = hist2d(x_tg, y_tg, bins=5)
    h_tg = h_tg.realize()

    assert_close(h_tg, Tensor(h_np))


@assert_one_kernel
def test_bins_list():
    x_np = np.random.rand(100)
    y_np = np.random.rand(100)
    x_tg = Tensor(x_np).realize()
    y_tg = Tensor(y_np).realize()

    h_np, _, _ = np.histogram2d(x_np, y_np, bins=[5, 10])
    h_tg, _, _ = hist2d(x_tg, y_tg, bins=[5, 10])
    h_tg = h_tg.realize()

    assert_close(h_tg, Tensor(h_np))


@assert_one_kernel
def test_range():
    x_np = np.array([-1, 0, 1, 2, 3, 4])
    y_np = np.array([-1, 0, 1, 2, 3, 4])
    x_tg = Tensor(x_np).realize()
    y_tg = Tensor(y_np).realize()

    range_np = [[0, 3], [0, 3]]

    h_np, _, _ = np.histogram2d(x_np, y_np, bins=3, range=range_np)
    h_tg, _, _ = hist2d(x_tg, y_tg, bins=3, range=range_np)
    h_tg = h_tg.realize()

    assert_close(h_tg, Tensor(h_np))


@assert_one_kernel
def test_density():
    x_np = np.random.rand(100)
    y_np = np.random.rand(100)
    x_tg = Tensor(x_np).realize()
    y_tg = Tensor(y_np).realize()

    h_np, _, _ = np.histogram2d(x_np, y_np, bins=5, density=True)
    h_tg, _, _ = hist2d(x_tg, y_tg, bins=5, density=True)
    h_tg = h_tg.realize()

    assert_close(h_tg, Tensor(h_np), atol=1e-6, rtol=1e-6)
