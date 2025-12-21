from tinygrad import Tensor

def flip(x: Tensor, flip_code: int) -> Tensor:
    """
    Flips an image horizontally, vertically, or both.
    Args:
        x: Input tensor with shape (H, W, C).
        flip_code:
            0: Flip vertically (around the x-axis).
            1: Flip horizontally (around the y-axis).
           -1: Flip both horizontally and vertically.
    Returns:
        Flipped tensor.
    """
    if flip_code == 0:
        return x.flip(0)
    elif flip_code == 1:
        return x.flip(1)
    elif flip_code == -1:
        return x.flip(0).flip(1)
    else:
        raise ValueError("flip_code must be 0, 1, or -1")
