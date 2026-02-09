import numpy as np
import pytest
from filterpy.kalman import MerweScaledSigmaPoints
from tinygrad import Tensor

from tinyops._core import assert_close, assert_one_kernel
from tinyops.signal.sigma_points import sigma_points

TEST_PARAMS = [
    (2, 1e-3, 2.0, 0.0, np.array([1.0, 2.0]), np.array([[2.0, 0.5], [0.5, 2.0]])),
    (3, 0.1, 2.0, 1.0, np.array([1.0, 2.0, 3.0]), np.eye(3) * 0.5),
]


@pytest.mark.parametrize("n,alpha,beta,kappa,x_np,P_np", TEST_PARAMS)
@assert_one_kernel
def test_sigma_points(n, alpha, beta, kappa, x_np, P_np):
    merwe = MerweScaledSigmaPoints(n, alpha, beta, kappa)
    sigmas_expected = merwe.sigma_points(x_np, P_np)
    Wm_expected = merwe.Wm
    Wc_expected = merwe.Wc

    x = Tensor(x_np).realize()
    P = Tensor(P_np).realize()

    sigmas, Wm, Wc = sigma_points(x, P, alpha, beta, kappa)
    sigmas_r = sigmas.realize()
    Wm_r = Wm.realize()
    Wc_r = Wc.realize()

    assert_close(sigmas_r, sigmas_expected)
    assert_close(Wm_r, Wm_expected)
    assert_close(Wc_r, Wc_expected)
