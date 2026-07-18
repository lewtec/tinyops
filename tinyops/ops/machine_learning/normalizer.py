from enum import Enum

from tinygrad import Tensor

from tinyops.ops.machine_learning._scaling import replace_zero_scale_with_one


class NormType(Enum):
    """Normalization types for the normalizer."""

    L1 = "l1"
    L2 = "l2"
    MAX = "max"


def normalizer(
    features: Tensor,
    norm_type: NormType = NormType.L2,
    axis: int = 1,
) -> Tensor:
    """Normalize samples individually to unit norm.

    Args:
        features: Input tensor.
        norm_type: Type of norm to use (L1, L2, or MAX).
        axis: Axis along which to normalize.

    Returns:
        Normalized tensor.
    """
    if norm_type == NormType.L1:
        norms = features.abs().sum(axis=axis, keepdim=True)
    elif norm_type == NormType.L2:
        norms = features.pow(2).sum(axis=axis, keepdim=True).sqrt()
    elif norm_type == NormType.MAX:
        norms = features.abs().max(axis=axis, keepdim=True)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")

    safe_norms = replace_zero_scale_with_one(norms)
    return features / safe_norms
