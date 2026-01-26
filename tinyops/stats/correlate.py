from typing import Literal

from tinygrad import Tensor


def correlate(a: Tensor, v: Tensor, mode: Literal["valid", "same", "full"] = "valid") -> Tensor:
    """
    Cross-correlation of two 1-dimensional sequences.
    """
    if a.ndim != 1 or v.ndim != 1:
        raise ValueError("a and v must be 1-dimensional")

    n, m = a.shape[0], v.shape[0]

    if mode == "valid":
        if n < m:
            return Tensor([])
        output_len = n - m + 1
        padded_a = a
    elif mode == "same":
        output_len = n
        pad_left = (m - 1) // 2
        pad_right = m - 1 - pad_left
        padded_a = Tensor.cat(Tensor.zeros(pad_left), a, Tensor.zeros(pad_right))
    elif mode == "full":
        output_len = n + m - 1
        pad = m - 1
        padded_a = Tensor.cat(Tensor.zeros(pad), a, Tensor.zeros(pad))

    result = []
    for i in range(output_len):
        result.append((padded_a[i : i + m] * v).sum())

    if not result:
        return Tensor([])

    return Tensor.stack(result)
