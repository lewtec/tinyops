from tinygrad import Tensor


def pairwise_hamming_distance(samples: Tensor) -> Tensor:
    """Compute pairwise Hamming distances between all samples.

    Args:
        samples: Input tensor (n_samples, n_features).

    Returns:
        Distance matrix (n_samples, n_samples).
    """
    return (samples.unsqueeze(1) != samples.unsqueeze(0)).sum(axis=2) / samples.shape[1]
