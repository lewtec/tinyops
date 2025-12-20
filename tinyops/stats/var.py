from tinygrad import Tensor
from typing import Optional, Union, Tuple

def var(a: Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, ddof: int = 0, keepdims: bool = False) -> Tensor:
    """
    Compute the variance along the specified axis.
    """
    return a.var(axis=axis, correction=ddof, keepdim=keepdims)
