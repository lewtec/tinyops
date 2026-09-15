from tinygrad import Tensor

from tinyops.ops._tensor_utils import unique_sorted_values


def label_encoder(labels: Tensor) -> Tensor:
    """Encode target labels as integers.

    Maps each unique value to a contiguous integer starting from 0.

    Args:
        labels: Input label tensor.

    Returns:
        Integer-encoded label tensor.
    """
    unique_labels = Tensor(unique_sorted_values(labels), requires_grad=False, device=labels.device)
    comparison = labels.unsqueeze(1) == unique_labels.unsqueeze(0)
    encoded = comparison.argmax(axis=1)
    return encoded.cast(labels.dtype)
