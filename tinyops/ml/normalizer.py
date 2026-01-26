from tinygrad import Tensor


def normalizer(X: Tensor, norm: str = "l2", axis: int = 1) -> Tensor:
    if norm not in ("l1", "l2", "max"):
        raise ValueError(f"Unsupported norm: {norm}")

    if norm == "l1":
        norms = X.abs().sum(axis=axis, keepdim=True)
    elif norm == "l2":
        norms = X.pow(2).sum(axis=axis, keepdim=True).sqrt()
    else:  # max
        norms = X.abs().max(axis=axis, keepdim=True)

    # Avoid division by zero
    norms = Tensor.where(norms == 0, 1.0, norms)

    return X / norms
