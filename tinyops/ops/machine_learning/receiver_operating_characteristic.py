from tinygrad import Tensor

from tinyops.ops._tensor_utils import unique_sorted_values


def receiver_operating_characteristic_area(
    true_labels: Tensor,
    prediction_scores: Tensor,
) -> Tensor:
    """Compute the Area Under the ROC Curve (AUC) from prediction scores.

    Uses the Mann-Whitney U statistic formulation.

    Args:
        true_labels: Binary ground truth labels (0 or 1).
        prediction_scores: Predicted scores or probabilities.

    Returns:
        AUC score as a scalar tensor.

    Raises:
        ValueError: If labels are not binary or only one class is present.
    """
    unique_labels = unique_sorted_values(true_labels)
    if len(unique_labels) != 2:
        raise ValueError("ROC AUC score is only defined for binary classification.")
    if 1 not in unique_labels or 0 not in unique_labels:
        raise ValueError("Only one class present in true_labels.")

    positive_mask = true_labels == 1
    negative_mask = true_labels == 0

    positive_count = positive_mask.sum()
    negative_count = negative_mask.sum()

    scores_i = prediction_scores.unsqueeze(1)
    scores_j = prediction_scores.unsqueeze(0)

    positive_i = positive_mask.unsqueeze(1)
    negative_j = negative_mask.unsqueeze(0)

    pair_mask = positive_i * negative_j
    concordant = (scores_i > scores_j) * pair_mask
    tied = (scores_i == scores_j) * pair_mask

    auc_sum = concordant.sum() + 0.5 * tied.sum()
    return auc_sum / (positive_count * negative_count)
