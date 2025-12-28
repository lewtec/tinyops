import pytest
from tinygrad import Tensor
from tinyops.test_utils import assert_one_kernel

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
