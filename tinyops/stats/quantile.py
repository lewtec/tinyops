from tinygrad import Tensor
from typing import Optional, Union, Tuple, List
from .percentile import percentile

def quantile(a: Tensor, q: Union[float, List[float], Tensor], axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False, method: str = 'linear') -> Tensor:
    """
    Compute the q-th quantile of the data along the specified axis.
    """
    if isinstance(q, Tensor):
        q_pct = q * 100.0
    elif isinstance(q, list):
        q_pct = [x * 100.0 for x in q]
    else:
        q_pct = q * 100.0

    return percentile(a, q_pct, axis=axis, keepdims=keepdims, method=method)
