from enum import Enum
from functools import partial
from tinygrad import Tensor

def rotate_90_clockwise(x: Tensor) -> Tensor:
  if x.ndim == 2:
    permute_order = (1, 0)
  elif x.ndim == 3:
    permute_order = (1, 0, 2)
  else:
    raise ValueError(f"Unsupported tensor rank: {x.ndim}")
  return x.permute(permute_order).flip(1)

def rotate_180(x: Tensor) -> Tensor:
  return x.flip((0, 1))

def rotate_90_counterclockwise(x: Tensor) -> Tensor:
  if x.ndim == 2:
    permute_order = (1, 0)
  elif x.ndim == 3:
    permute_order = (1, 0, 2)
  else:
    raise ValueError(f"Unsupported tensor rank: {x.ndim}")
  return x.permute(permute_order).flip(0)

class RotateCode(Enum):
  CLOCKWISE_90 = partial(rotate_90_clockwise)
  ROTATE_180 = partial(rotate_180)
  COUNTERCLOCKWISE_90 = partial(rotate_90_counterclockwise)

  def __call__(self, *args, **kwargs):
    return self.value(*args, **kwargs)

# Backward compatibility constants
ROTATE_90_CLOCKWISE = 0
ROTATE_180 = 1
ROTATE_90_COUNTERCLOCKWISE = 2

_INT_TO_ROTATE_CODE = {
    ROTATE_90_CLOCKWISE: RotateCode.CLOCKWISE_90,
    ROTATE_180: RotateCode.ROTATE_180,
    ROTATE_90_COUNTERCLOCKWISE: RotateCode.COUNTERCLOCKWISE_90
}

def rotate(x: Tensor, rotate_code: int | RotateCode) -> Tensor:
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
  if isinstance(rotate_code, int):
    if rotate_code in _INT_TO_ROTATE_CODE:
        code = _INT_TO_ROTATE_CODE[rotate_code]
    else:
        raise ValueError("Invalid rotate_code")
  elif isinstance(rotate_code, RotateCode):
    code = rotate_code
  else:
    raise TypeError(f"Invalid type for rotate_code: {type(rotate_code)}")

  return code(x)
