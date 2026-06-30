from tinygrad import Tensor

def calculate_binary_components(true_labels: Tensor, predicted_labels: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Calculate the core binary classification components.

    Args:
        true_labels: Ground truth binary labels.
        predicted_labels: Predicted binary labels.

    Returns:
        A tuple containing (true_positives, predicted_positives, actual_positives).
    """
    true_positives = (true_labels * predicted_labels).sum()
    predicted_positives = predicted_labels.sum()
    actual_positives = true_labels.sum()
    return true_positives, predicted_positives, actual_positives
