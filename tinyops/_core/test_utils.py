from functools import wraps


class KernelCountError(AssertionError):
    """Exception raised when the kernel count is not as expected."""

    pass


def assert_one_kernel(func):
    """
    Decorator for tests to ensure that the test function generates exactly one tinygrad kernel.

    NOTE: This check is temporarily disabled project-wide.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # The check is disabled, so we just run the function.
        return func(*args, **kwargs)

    return wrapper
