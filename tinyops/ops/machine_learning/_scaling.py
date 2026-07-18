"""Shared helpers for feature scaling operations."""

from tinygrad import Tensor


def replace_zero_scale_with_one(scale: Tensor) -> Tensor:
    """Replace zero scale factors with one so constant features stay unchanged.

    Feature scalers divide by a per-feature scale (std, range, IQR, max abs).
    When that scale is zero the feature is constant; returning the original
    values is achieved by dividing by one instead of zero.
    """
    return Tensor.where(scale == 0, 1.0, scale)
