from collections.abc import Sequence

from tinygrad import Tensor


def tensordot(a: Tensor, b: Tensor, axes: int | Sequence[int | Sequence[int]] = 2) -> Tensor:
    """Compute tensor dot product along specified axes."""
    axes_a: list[int]
    axes_b: list[int]
    if isinstance(axes, int):
        axes_a = list(range(len(a.shape) - axes, len(a.shape)))
        axes_b = list(range(0, axes))
    else:
        ax_a, ax_b = axes
        axes_a = [ax_a] if isinstance(ax_a, int) else list(ax_a)
        axes_b = [ax_b] if isinstance(ax_b, int) else list(ax_b)
    nda, ndb = len(a.shape), len(b.shape)
    axes_a = [ax + nda if ax < 0 else ax for ax in axes_a]
    axes_b = [ax + ndb if ax < 0 else ax for ax in axes_b]
    if len(axes_a) != len(axes_b):
        raise ValueError("Different number of axes")
    for i, j in zip(axes_a, axes_b):
        if a.shape[i] != b.shape[j]:
            raise ValueError(f"Shape mismatch: {a.shape[i]} != {b.shape[j]}")
    free_a = [i for i in range(nda) if i not in axes_a]
    free_b = [i for i in range(ndb) if i not in axes_b]
    new_a = a.permute(free_a + axes_a)
    new_b = b.permute(axes_b + free_b)
    prod_free_a = 1
    for i in free_a:
        prod_free_a *= a.shape[i]
    prod_contract = 1
    for i in axes_a:
        prod_contract *= a.shape[i]
    prod_free_b = 1
    for i in free_b:
        prod_free_b *= b.shape[i]
    flat_a = new_a.reshape(prod_free_a, prod_contract)
    flat_b = new_b.reshape(prod_contract, prod_free_b)
    res = flat_a.matmul(flat_b)
    return res.reshape([a.shape[i] for i in free_a] + [b.shape[i] for i in free_b])
