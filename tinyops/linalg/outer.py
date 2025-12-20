from tinygrad import Tensor

def outer(a: Tensor, b: Tensor) -> Tensor:
    """
    Compute the outer product of two vectors.
    """
    a_flat = a.flatten()
    b_flat = b.flatten()
    return a_flat.unsqueeze(1) * b_flat.unsqueeze(0)
