from tinygrad import Tensor
from typing import Optional, Union, Tuple

def std(a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, ddof: int = 0, keepdims: bool = False) -> Tensor:
    """
    Compute the standard deviation along the specified axis.
    """
    return a.std(axis=axis, correction=ddof, keepdim=keepdims)
