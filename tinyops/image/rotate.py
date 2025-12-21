from tinygrad import Tensor

ROTATE_90_CLOCKWISE = 0
ROTATE_180 = 1
ROTATE_90_COUNTERCLOCKWISE = 2

def rotate(x: Tensor, rotate_code: int) -> Tensor:
  """
  Rotates a 2D array in multiples of 90 degrees.

  Args:
    x: Input tensor.
    rotate_code: A flag to specify how to rotate the array.
      - 0: Rotate 90 degrees clockwise.
      - 1: Rotate 180 degrees.
      - 2: Rotate 90 degrees counter-clockwise.

  Returns:
    A new tensor with the rotated image.
  """
  if x.ndim == 2:
    permute_order = (1, 0)
  elif x.ndim == 3:
    permute_order = (1, 0, 2)
  else:
    raise ValueError(f"Unsupported tensor rank: {x.ndim}")

  if rotate_code == ROTATE_90_CLOCKWISE:
    return x.permute(permute_order).flip(1)
  elif rotate_code == ROTATE_180:
    return x.flip((0, 1))
  elif rotate_code == ROTATE_90_COUNTERCLOCKWISE:
    return x.permute(permute_order).flip(0)
  else:
    raise ValueError("Invalid rotate_code")
