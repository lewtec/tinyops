from tinygrad import Tensor
from typing import Optional, Union, Tuple

def mean(a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    """
    Compute the arithmetic mean along the specified axis.
    """
    return a.mean(axis=axis, keepdim=keepdims)
