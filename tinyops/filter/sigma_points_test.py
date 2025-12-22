import numpy as np
import pytest
from tinygrad import Tensor
from tinyops.filter.sigma_points import sigma_points
from filterpy.kalman import MerweScaledSigmaPoints
from tinyops._core import assert_close

def test_sigma_points_2d():
    n = 2
    alpha = 1e-3
    beta = 2.0
    kappa = 0.0

    x_np = np.array([1.0, 2.0])
    P_np = np.array([[2.0, 0.5], [0.5, 2.0]])

    merwe = MerweScaledSigmaPoints(n, alpha, beta, kappa)
    sigmas_expected = merwe.sigma_points(x_np, P_np)
    Wm_expected = merwe.Wm
    Wc_expected = merwe.Wc

    x = Tensor(x_np)
    P = Tensor(P_np)

    sigmas, Wm, Wc = sigma_points(x, P, alpha, beta, kappa)

    assert_close(sigmas, sigmas_expected)
    assert_close(Wm, Wm_expected)
    assert_close(Wc, Wc_expected)

def test_sigma_points_3d():
    n = 3
    alpha = 0.1
    beta = 2.0
    kappa = 1.0

    x_np = np.array([1.0, 2.0, 3.0])
    P_np = np.eye(3) * 0.5

    merwe = MerweScaledSigmaPoints(n, alpha, beta, kappa)
    sigmas_expected = merwe.sigma_points(x_np, P_np)
    Wm_expected = merwe.Wm
    Wc_expected = merwe.Wc

    x = Tensor(x_np)
    P = Tensor(P_np)

    sigmas, Wm, Wc = sigma_points(x, P, alpha, beta, kappa)

    assert_close(sigmas, sigmas_expected)
    assert_close(Wm, Wm_expected)
    assert_close(Wc, Wc_expected)
