from tinygrad import Tensor

def pairwise_hamming_distance(X: Tensor) -> Tensor:
    """
    Compute the pairwise Hamming distance between samples in X.

    Args:
        X: A tinygrad Tensor of shape (n_samples, n_features).

    Returns:
        A tinygrad Tensor of shape (n_samples, n_samples) with the pairwise distances.
    """
    return (X.unsqueeze(1) != X.unsqueeze(0)).sum(axis=2) / X.shape[1]
