import pytest
from tinygrad import Tensor
from tinyops.test_utils import assert_one_kernel

# Helper fixture to provide realized tensors, avoiding kernel counts during setup
@pytest.fixture
def realized_inputs():
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    a.realize()
    b.realize()
    return a, b

def test_assert_one_kernel_passes_with_one_kernel(realized_inputs):
    """Test that the decorator passes when exactly one kernel is generated."""

    @assert_one_kernel
    def valid_test():
        a, b = realized_inputs
        c = a + b
        c.realize()

    valid_test()

def test_assert_one_kernel_fails_with_zero_kernels():
    """Test that the decorator raises AssertionError when no kernels are generated."""

    @assert_one_kernel
    def no_kernel_test():
        # Only lazy definition, no realization
        a = Tensor([1])
        b = Tensor([2])
        c = a + b

    with pytest.raises(AssertionError, match="Expected exactly 1 kernel, but got 0"):
        no_kernel_test()

def test_assert_one_kernel_fails_with_multiple_kernels(realized_inputs):
    """Test that the decorator raises AssertionError when multiple kernels are generated."""

    @assert_one_kernel
    def multi_kernel_test():
        a, b = realized_inputs
        # First kernel
        c = a + b
        c.realize()

        # Second kernel
        d = c * 2
        d.realize()

    with pytest.raises(AssertionError, match="Expected exactly 1 kernel, but got 2"):
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

    # Expect failure because a, b creation/loading likely adds kernels
    # Based on earlier probe: List setup + op was 3 (load, load, add) -> wait, probe said 2 or 3 depending on structure
    # Probe said: "List + Add count: 2" for one list load + add.
    # Here we have two list loads.
    with pytest.raises(AssertionError) as excinfo:
        test_with_internal_setup()

    assert "Expected exactly 1 kernel" in str(excinfo.value)
