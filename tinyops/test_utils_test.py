import pytest
from tinygrad import Tensor
from tinyops.test_utils import assert_one_kernel, KernelCountError

# Helper to ensure realization happens before test
def get_realized_tensors():
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    a.realize()
    b.realize()
    return a, b

@pytest.mark.parametrize("unused_arg", [None])
def test_assert_one_kernel_passes_with_one_kernel(unused_arg):
    """Test that the decorator passes when exactly one kernel is generated."""
    # Setup inputs
    a, b = get_realized_tensors()

    @assert_one_kernel
    def valid_test():
        c = a + b
        c.realize()

    valid_test()

@pytest.mark.parametrize("val1, val2", [(1, 2)])
def test_assert_one_kernel_fails_with_zero_kernels(val1, val2):
    """Test that the decorator raises KernelCountError when no kernels are generated."""

    @assert_one_kernel
    def no_kernel_test():
        # Only lazy definition, no realization
        a = Tensor([val1])
        b = Tensor([val2])
        c = a + b

    with pytest.raises(KernelCountError):
        no_kernel_test()

@pytest.mark.parametrize("factor", [2])
def test_assert_one_kernel_fails_with_multiple_kernels(factor):
    """Test that the decorator raises KernelCountError when multiple kernels are generated."""
    a, b = get_realized_tensors()

    @assert_one_kernel
    def multi_kernel_test():
        # First kernel
        c = a + b
        c.realize()

        # Second kernel
        d = c * factor
        d.realize()

    with pytest.raises(KernelCountError):
        multi_kernel_test()

@pytest.mark.parametrize("data", [[1, 2, 3]])
def test_assert_one_kernel_fails_with_setup_kernels(data):
    """Test showing that setup inside the function counts towards the limit."""

    @assert_one_kernel
    def test_with_internal_setup():
        # Creating from list generates a LoadOp/kernel usually
        a = Tensor(data)
        b = Tensor(data)
        # Even if we realize here, it counts
        c = a + b
        c.realize()

    with pytest.raises(KernelCountError):
        test_with_internal_setup()
