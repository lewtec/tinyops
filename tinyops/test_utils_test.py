import pytest
from tinygrad import Tensor
from tinyops.test_utils import assert_one_kernel, KernelCountError

# Utility to create realized tensors without fixtures
def get_realized_tensors():
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    a.realize()
    b.realize()
    return a, b

def test_assert_one_kernel_passes_with_one_kernel():
    """Test that the decorator passes when exactly one kernel is generated."""
    # Pre-calculate inputs to avoid setup kernels inside the decorated function
    a, b = get_realized_tensors()

    @assert_one_kernel
    def valid_test():
        c = a + b
        c.realize()

    valid_test()

def test_assert_one_kernel_fails_with_zero_kernels():
    """Test that the decorator raises KernelCountError when no kernels are generated."""

    @assert_one_kernel
    def no_kernel_test():
        # Only lazy definition, no realization
        a = Tensor([1])
        b = Tensor([2])
        c = a + b

    with pytest.raises(KernelCountError):
        no_kernel_test()

def test_assert_one_kernel_fails_with_multiple_kernels():
    """Test that the decorator raises KernelCountError when multiple kernels are generated."""
    a, b = get_realized_tensors()

    @assert_one_kernel
    def multi_kernel_test():
        # First kernel
        c = a + b
        c.realize()

        # Second kernel
        d = c * 2
        d.realize()

    with pytest.raises(KernelCountError):
        multi_kernel_test()

def test_assert_one_kernel_fails_with_setup_kernels():
    """Test showing that setup inside the function counts towards the limit."""

    @assert_one_kernel
    def test_with_internal_setup():
        # Creating from list generates a LoadOp/kernel usually
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        # Even if we realize here, it counts
        c = a + b
        c.realize()

    with pytest.raises(KernelCountError):
        test_with_internal_setup()
