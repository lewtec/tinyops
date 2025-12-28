from tinygrad import Tensor

def hamming_distance(u: Tensor, v: Tensor) -> Tensor:
  """
  Computes the Hamming distance between two 1-D tensors.

  The Hamming distance between two vectors, `u` and `v`, is the
  proportion of components in which `u[i] != v[i]`.

  Args:
    u: The first 1-D tensor.
    v: The second 1-D tensor.

  Returns:
    A 0-D tensor representing the Hamming distance.
  """
  if len(u.shape) != 1 or len(v.shape) != 1:
    raise ValueError("Input tensors must be 1-D.")
  if u.shape[0] != v.shape[0]:
    raise ValueError("Input tensors must have the same length.")

  return (u != v).float().mean()
