from tinygrad import Tensor
from typing import Optional

def bincount(x: Tensor, weights: Optional[Tensor] = None, minlength: int = 0) -> Tensor:
    """
    Count occurrence of each value in array of non-negative integers.
    """
    if x.ndim != 1:
        raise ValueError("x must be 1D")

    if x.numel() == 0:
        return Tensor.zeros(minlength)

    min_val = int(x.min().item())
    if min_val < 0:
        raise ValueError("Input must be non-negative")

    max_val = int(x.max().item())

    num_classes = max(max_val + 1, minlength)

    oh = x.one_hot(num_classes)

    if weights is None:
        return oh.sum(0)
    else:
        if weights.shape != x.shape:
            raise ValueError("weights and x must have the same shape")
        return (oh * weights.unsqueeze(-1)).sum(0)
