from tinygrad import Tensor, dtypes

# Define a reasonable limit for the number of samples to prevent DoS.
MAX_SAMPLES = 2048

def nearest_neighbors(X: Tensor, n_neighbors: int) -> Tensor:
    """
    Finds the K-Nearest Neighbors for each point in the input tensor.

    Args:
        X (Tensor): The input tensor of shape (n_samples, n_features).
        n_neighbors (int): The number of neighbors to find.

    Returns:
        Tensor: A tensor of shape (n_samples, n_neighbors) containing the indices of the nearest neighbors.
    """
    if X.shape[0] > MAX_SAMPLES:
        raise ValueError(f"Number of samples ({X.shape[0]}) exceeds the maximum limit of {MAX_SAMPLES} to prevent potential DoS.")

    # Iterative approach to avoid large intermediate tensors that may be causing crashes.
    neighbor_indices = []
    for i in range(X.shape[0]):
        # Calculate squared Euclidean distance from the i-th point to all other points
        point = X[i].unsqueeze(0)
        dists_sq = (point - X).pow(2).sum(axis=1)

        # Sort distances and get the indices of the n_neighbors smallest distances
        _, indices = dists_sq.sort()
        neighbor_indices.append(indices[0:n_neighbors].unsqueeze(0))

    # Concatenate the results for all points and ensure the output is integer.
    return Tensor.cat(*neighbor_indices, dim=0).cast(dtypes.int32)
