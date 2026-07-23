from tinygrad import Tensor, dtypes

from tinyops.ops._tensor_utils import unique_sorted_values


def label_encoder(labels: Tensor) -> Tensor:
    """Encode target labels as integers.

    Maps each unique value to a contiguous integer starting from 0,
    ordered by sorted unique values (matching sklearn.preprocessing.LabelEncoder).

    Args:
        labels: Input label tensor (numeric categories).

    Returns:
        Integer-encoded label tensor (int64 codes).
    """
    unique_labels = Tensor(unique_sorted_values(labels), requires_grad=False, device=labels.device)
    comparison = labels.unsqueeze(1) == unique_labels.unsqueeze(0)
    encoded = comparison.argmax(axis=1)
    # sklearn always returns integer codes regardless of input dtype.
    return encoded.cast(dtypes.int64)
