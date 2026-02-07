from tinygrad import Tensor


def roc_auc(y_true: Tensor, y_score: Tensor) -> Tensor:
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
    """
    # Validation using tinygrad operations.
    # Check if only binary labels (0 or 1) are present.
    # Note: This check assumes labels are 0 and 1. The original implementation relied on unique count being 2,
    # but also hardcoded pos_mask = y_true == 1, implying 1 is the positive class.

    # We check if there are any values that are neither 0 nor 1.
    # If y_true contains values other than 0 and 1, ((y_true != 0) * (y_true != 1)) will be non-zero at those positions.
    if ((y_true != 0) * (y_true != 1)).sum().item() > 0:
         raise ValueError("ROC AUC score is only defined for binary classification.")

    # Check if both classes are present.
    n_pos = (y_true == 1).sum().item()
    n_neg = (y_true == 0).sum().item()

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Only one class present in y_true. ROC AUC score is not defined in that case.")

    # Main computation using tinygrad broadcasting
    pos_mask = y_true == 1
    neg_mask = y_true == 0

    # Broadcast to compare every score with every other score.
    y_score_i = y_score.unsqueeze(1)
    y_score_j = y_score.unsqueeze(0)

    pos_mask_i = pos_mask.unsqueeze(1)
    neg_mask_j = neg_mask.unsqueeze(0)

    # Create a mask for pairs of (positive, negative) samples.
    pair_mask = pos_mask_i * neg_mask_j

    # Count pairs where positive score is greater than negative score.
    greater_mask = y_score_i > y_score_j
    auc_sum = (greater_mask * pair_mask).sum()

    # Count pairs where scores are equal.
    equal_mask = y_score_i == y_score_j
    auc_sum = auc_sum + 0.5 * (equal_mask * pair_mask).sum()

    # Normalize by the total number of pairs.
    auc = auc_sum / (n_pos * n_neg)

    return auc
