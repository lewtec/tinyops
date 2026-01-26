from tinygrad import Tensor


def vdot(a: Tensor, b: Tensor) -> Tensor:
    """
    Return the dot product of two vectors.
    """
    return (a.flatten() * b.flatten()).sum()
