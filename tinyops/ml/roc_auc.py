from tinygrad import Tensor


def roc_auc(y_true: Tensor, y_score: Tensor) -> Tensor:
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
    """
    import numpy as np

    # Validation using numpy. This is fine as it happens before the graph computation.
    y_true_np = y_true.numpy()
    unique_labels = np.unique(y_true_np)
    if len(unique_labels) != 2:
        raise ValueError("ROC AUC score is only defined for binary classification.")
    if np.sum(y_true_np == 1) == 0 or np.sum(y_true_np == 0) == 0:
        raise ValueError("Only one class present in y_true. ROC AUC score is not defined in that case.")

    # Main computation using tinygrad broadcasting
    pos_mask = y_true == 1
    neg_mask = y_true == 0

    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

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
