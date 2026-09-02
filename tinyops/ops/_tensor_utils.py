"""Shared tensor helpers for ops that must stay on tinygrad + stdlib only."""

from collections.abc import Sequence

from tinygrad import Tensor


def unique_sorted_values(values: Tensor) -> list:
    """Return sorted unique scalar values from a tensor (stdlib only)."""
    flat = values.reshape(-1).tolist()
    if not isinstance(flat, Sequence) or isinstance(flat, (str, bytes)):
        flat = [flat]
    return sorted(set(flat))
