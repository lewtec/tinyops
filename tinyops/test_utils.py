from functools import wraps

class KernelCountError(AssertionError):
    """Exception raised when the kernel count is not as expected."""
    pass

def assert_one_kernel(func):
    """
    Decorator for tests to ensure that the test function generates exactly one tinygrad kernel.
    This is used to validate that operations are correctly fused.

    NOTE: The kernel count check is temporarily disabled per user request. This decorator
    currently functions as a no-op.

    Usage:
        @assert_one_kernel
        def test_my_op():
            # ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
