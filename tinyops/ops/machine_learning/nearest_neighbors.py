from tinygrad import Tensor, dtypes


def nearest_neighbors(samples: Tensor, neighbor_count: int) -> Tensor:
    """Find the k-nearest neighbors for each sample.

    Args:
        samples: Input tensor of shape (n_samples, n_features).
        neighbor_count: Number of neighbors to find.

    Returns:
        Tensor of shape (n_samples, neighbor_count) containing neighbor indices.
    """
    neighbor_indices = []
    for sample_index in range(samples.shape[0]):
        point = samples[sample_index].unsqueeze(0)
        squared_distances = (point - samples).pow(2).sum(axis=1)
        _, indices = squared_distances.sort()
        neighbor_indices.append(indices[0:neighbor_count].unsqueeze(0))

    return Tensor.cat(*neighbor_indices, dim=0).cast(dtypes.int32)
