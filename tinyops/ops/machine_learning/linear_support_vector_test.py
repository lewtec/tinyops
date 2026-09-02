"""Pure-tinygrad tests for shared linear support-vector decision logic."""

from tinygrad import Tensor

from tinyops.ops.machine_learning._linear_support_vector import _linear_support_vector_decision
from tinyops.ops.machine_learning.linear_support_vector_classifier import (
    linear_support_vector_classifier,
)
from tinyops.ops.machine_learning.linear_support_vector_regressor import (
    linear_support_vector_regressor,
)


def test_linear_support_vector_decision_matches_hand_computed():
    samples = Tensor([[1.0, 2.0], [3.0, 4.0], [0.0, 1.0]])
    coefficients = Tensor([[0.5, -1.0], [2.0, 0.0]])
    intercept = Tensor([0.25, -0.5])

    got = _linear_support_vector_decision(samples, coefficients, intercept).numpy()
    # row i: samples[i] @ coefficients.T + intercept
    expected = [
        [1.0 * 0.5 + 2.0 * -1.0 + 0.25, 1.0 * 2.0 + 2.0 * 0.0 - 0.5],
        [3.0 * 0.5 + 4.0 * -1.0 + 0.25, 3.0 * 2.0 + 4.0 * 0.0 - 0.5],
        [0.0 * 0.5 + 1.0 * -1.0 + 0.25, 0.0 * 2.0 + 1.0 * 0.0 - 0.5],
    ]
    assert got.shape == (3, 2)
    for row_got, row_exp in zip(got, expected, strict=True):
        for value_got, value_exp in zip(row_got, row_exp, strict=True):
            assert abs(float(value_got) - value_exp) < 1e-6


def test_classifier_and_regressor_share_decision_surface():
    samples = Tensor([[1.0, -1.0], [2.0, 3.0]])
    coefficients = Tensor([[1.5, -0.5]])
    intercept = Tensor([0.1])

    shared = _linear_support_vector_decision(samples, coefficients, intercept)
    classified = linear_support_vector_classifier(samples, coefficients, intercept)
    regressed = linear_support_vector_regressor(samples, coefficients, intercept)

    assert (classified - shared).abs().max().numpy() < 1e-6
    assert (regressed - shared.flatten()).abs().max().numpy() < 1e-6
    assert regressed.shape == (2,)
