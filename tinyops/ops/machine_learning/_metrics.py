from tinygrad import Tensor

def _true_positives(true_labels: Tensor, predicted_labels: Tensor) -> Tensor:
    """Calculate true positives for binary classification."""
    return (true_labels * predicted_labels).sum()


def _safe_divide(numerator: Tensor, denominator: Tensor) -> Tensor:
    """Safely divide two tensors, returning 0 where the denominator is 0."""
    return numerator / Tensor.where(denominator == 0, 1, denominator)
