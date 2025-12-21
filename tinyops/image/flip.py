from tinygrad import Tensor

def flip(x: Tensor, flip_code: int) -> Tensor:
  """
  Flips a 2D array around vertical, horizontal, or both axes.

  Args:
    x: Input tensor.
    flip_code: A flag to specify how to flip the array; 0 means
      flipping around the x-axis and positive value (e.g., 1) means
      flipping around the y-axis. Negative value (e.g., -1) means
      flipping around both axes.

  Returns:
    A new tensor with the flipped image.
  """
  if flip_code == 0:
    return x.flip(0)
  elif flip_code > 0:
    return x.flip(1)
  else:
    return x.flip((0, 1))
