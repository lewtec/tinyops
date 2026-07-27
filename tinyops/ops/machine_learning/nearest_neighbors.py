from tinygrad import Tensor, dtypes


def nearest_neighbors(
    query_samples: Tensor,
    reference_samples: Tensor,
    neighbor_count: int,
) -> tuple[Tensor, Tensor]:
    """Find the k-nearest neighbors of each query among reference samples.

    Uses Euclidean distance (L2).

    Args:
        query_samples: Query points of shape (n_queries, n_features).
        reference_samples: Reference points of shape (n_references, n_features).
        neighbor_count: Number of neighbors to find per query.

    Returns:
        Tuple of:
            - distances: (n_queries, neighbor_count) Euclidean distances
            - indices: (n_queries, neighbor_count) integer indices into reference_samples
    """
    if neighbor_count < 1:
        raise ValueError(f"neighbor_count must be >= 1, got {neighbor_count}")
    if reference_samples.shape[0] < neighbor_count:
        raise ValueError(
            f"Expected n_neighbors <= n_samples, but n_samples = {reference_samples.shape[0]}, "
            f"n_neighbors = {neighbor_count}"
        )

    distance_rows: list[Tensor] = []
    index_rows: list[Tensor] = []
    for sample_index in range(query_samples.shape[0]):
        point = query_samples[sample_index].unsqueeze(0)
        squared_distances = (point - reference_samples).pow(2).sum(axis=1)
        sorted_squared, indices = squared_distances.sort()
        top_squared = sorted_squared[0:neighbor_count]
        top_indices = indices[0:neighbor_count]
        distance_rows.append(top_squared.sqrt().unsqueeze(0))
        index_rows.append(top_indices.unsqueeze(0))

    distances = Tensor.cat(*distance_rows, dim=0)
    indices = Tensor.cat(*index_rows, dim=0).cast(dtypes.int32)
    return distances, indices
