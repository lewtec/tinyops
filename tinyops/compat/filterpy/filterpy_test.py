"""Tests for FilterPy compatibility layer.

Compares tinyops.compat.filterpy against actual filterpy.
"""

import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints as FPMerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise as fp_Q_discrete_white_noise
from tinygrad import Tensor

from tinyops._core import assert_close
from tinyops.compat import filterpy as tfp


class TestQDiscreteWhiteNoise:
    def test_dim2(self):
        result = tfp.common.Q_discrete_white_noise(dim=2, dt=1.0, var=1.0)
        expected = fp_Q_discrete_white_noise(dim=2, dt=1.0, var=1.0)
        assert_close(result, np.array(expected, dtype=np.float32), atol=1e-4)

    def test_dim3(self):
        result = tfp.common.Q_discrete_white_noise(dim=3, dt=0.5, var=2.0)
        expected = fp_Q_discrete_white_noise(dim=3, dt=0.5, var=2.0)
        assert_close(result, np.array(expected, dtype=np.float32), atol=1e-4)

    def test_dim4(self):
        result = tfp.common.Q_discrete_white_noise(dim=4, dt=1.0, var=1.0)
        expected = fp_Q_discrete_white_noise(dim=4, dt=1.0, var=1.0)
        assert_close(result, np.array(expected, dtype=np.float32), atol=1e-4)

    def test_block_size(self):
        result = tfp.common.Q_discrete_white_noise(dim=2, dt=1.0, var=1.0, block_size=2)
        expected = fp_Q_discrete_white_noise(dim=2, dt=1.0, var=1.0, block_size=2)
        assert_close(result, np.array(expected, dtype=np.float32), atol=1e-4)


class TestMerweScaledSigmaPoints:
    def test_sigma_points_shape(self):
        n = 3
        sp = tfp.kalman.MerweScaledSigmaPoints(n=n, alpha=0.1, beta=2.0, kappa=0.0)
        mean = Tensor.zeros(n)
        cov = Tensor.eye(n)
        points = sp.sigma_points(mean, cov)
        assert points.shape == (2 * n + 1, n)

    def test_sigma_points_values(self):
        n = 2
        alpha, beta, kappa = 0.1, 2.0, 0.0
        mean = np.array([1.0, 2.0], dtype=np.float32)
        cov = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        sp_ours = tfp.kalman.MerweScaledSigmaPoints(n=n, alpha=alpha, beta=beta, kappa=kappa)
        points_ours = sp_ours.sigma_points(Tensor(mean), Tensor(cov))

        sp_ref = FPMerweScaledSigmaPoints(n=n, alpha=alpha, beta=beta, kappa=kappa)
        points_ref = sp_ref.sigma_points(mean, cov)

        assert_close(points_ours, np.array(points_ref, dtype=np.float32), atol=1e-4)

    def test_weights(self):
        n = 3
        alpha, beta, kappa = 0.5, 2.0, 1.0

        sp_ours = tfp.kalman.MerweScaledSigmaPoints(n=n, alpha=alpha, beta=beta, kappa=kappa)
        wm_ours, wc_ours = sp_ours.weights()

        sp_ref = FPMerweScaledSigmaPoints(n=n, alpha=alpha, beta=beta, kappa=kappa)
        wm_ref = sp_ref.Wm
        wc_ref = sp_ref.Wc

        assert_close(wm_ours, np.array(wm_ref, dtype=np.float32), atol=1e-4)
        assert_close(wc_ours, np.array(wc_ref, dtype=np.float32), atol=1e-4)