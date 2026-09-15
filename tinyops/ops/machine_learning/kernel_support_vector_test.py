"""Direct tests for kernel SVM decision helpers (no reference libraries)."""

from tinygrad import Tensor

from tinyops.ops.machine_learning._svm_kernel import KernelType, _kernel_support_vector_decision
from tinyops.ops.machine_learning.kernel_support_vector_classifier import (
    kernel_support_vector_classifier,
)
from tinyops.ops.machine_learning.kernel_support_vector_regressor import (
    kernel_support_vector_regressor,
)


def test_kernel_svm_classifier_and_regressor_share_decision_path():
    samples = Tensor([[0.0, 0.0], [1.0, 1.0], [2.0, -1.0]])
    support_vectors = Tensor([[0.0, 0.0], [1.0, 0.0]])
    dual_coefficients = Tensor([[1.0, -0.5]])
    intercept = Tensor([0.25])

    classifier = kernel_support_vector_classifier(
        samples,
        support_vectors,
        dual_coefficients,
        intercept,
        kernel=KernelType.LINEAR,
    )
    regressor = kernel_support_vector_regressor(
        samples,
        support_vectors,
        dual_coefficients,
        intercept,
        kernel=KernelType.LINEAR,
    )
    shared = _kernel_support_vector_decision(
        samples,
        support_vectors,
        dual_coefficients,
        intercept,
        kernel=KernelType.LINEAR,
    )

    assert classifier.shape == (3, 1)
    assert regressor.shape == classifier.shape
    assert shared.shape == classifier.shape
    # Public entry points must stay identical to the shared helper.
    assert (classifier - shared).abs().max().numpy() < 1e-5
    assert (regressor - shared).abs().max().numpy() < 1e-5


def test_kernel_svm_rbf_is_finite():
    samples = Tensor([[0.0, 1.0], [1.0, 0.0]])
    support_vectors = Tensor([[0.5, 0.5]])
    dual_coefficients = Tensor([[1.0]])
    intercept = Tensor([0.0])

    result = kernel_support_vector_classifier(
        samples,
        support_vectors,
        dual_coefficients,
        intercept,
        kernel=KernelType.RADIAL_BASIS_FUNCTION,
        gamma=0.5,
    )
    values = result.numpy().reshape(-1)
    assert values.shape == (2,)
    assert all(abs(float(v)) < 1e6 for v in values)
