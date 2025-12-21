from tinygrad import Tensor
from typing import Optional, Union, Tuple

def ptp(a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Tensor:
    """
    Range of values (maximum - minimum) along an axis.
    """
    return a.max(axis=axis, keepdim=keepdims) - a.min(axis=axis, keepdim=keepdims)
