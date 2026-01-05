from tinygrad import Tensor

# A reasonable limit to prevent DoS from broadcasting a massive intermediate tensor.
# This prevents a (n_samples, n_samples, n_features) tensor from exhausting memory.
MAX_INTERMEDIATE_ELEMENTS = 500_000_000  # 500 million elements

def pairwise_hamming_distance(X: Tensor) -> Tensor:
    """
    Compute the pairwise Hamming distance between samples in X.

    Args:
        X: A tinygrad Tensor of shape (n_samples, n_features).

    Returns:
        A tinygrad Tensor of shape (n_samples, n_samples) with the pairwise distances.
    """
    n_samples, n_features = X.shape
    intermediate_elements = n_samples * n_samples * n_features

    # 🛡️ Sentinel: Add security check to prevent DoS attack.
    # A large input tensor could create a massive intermediate tensor via broadcasting,
    # leading to a memory allocation crash.
    if intermediate_elements > MAX_INTERMEDIATE_ELEMENTS:
        raise ValueError(
            f"Input tensor with shape {X.shape} would create a broadcasted tensor with "
            f"{intermediate_elements} elements, exceeding the security limit of {MAX_INTERMEDIATE_ELEMENTS}."
        )

    return (X.unsqueeze(1) != X.unsqueeze(0)).sum(axis=2) / X.shape[1]
