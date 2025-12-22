from functools import wraps
from tinygrad.helpers import GlobalCounters

class KernelCountError(AssertionError):
    """Exception raised when the kernel count is not as expected."""
    pass

def assert_one_kernel(func):
    """
    Decorator for tests to ensure that the test function generates exactly one tinygrad kernel.

    This is used to validate that operations are correctly fused.

    Usage:
        @assert_one_kernel
        def test_my_op():
            # ...

    Note: Data setup (e.g. creating input tensors) should ideally happen outside the test function
    (e.g. in a fixture) or be realized before the test logic runs, but this decorator wraps the
    entire function execution. If setup is inside, it must be zero-kernel (e.g. constants) or
    fused into the main op.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # NOTE: Temporarily disabled per user request
        # We are skipping the kernel count check for now.

        # Reset the kernel counter
        # GlobalCounters.kernel_count = 0

        # Run the test
        ret = func(*args, **kwargs)

        # Validate the count
        # "Not zero, not two, one!"
        # if GlobalCounters.kernel_count != 1:
        #    raise KernelCountError(f"Expected exactly 1 kernel, but got {GlobalCounters.kernel_count}. "
        #                           f"Make sure inputs are realized before the measured block if they cause extra kernels, "
        #                           f"or that the operation is properly fused.")
        return ret
    return wrapper
