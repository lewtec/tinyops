"""SciPy compatibility layer.

Provides scipy-compatible function signatures that delegate to tinyops.ops.
"""


class _SpatialDistance:
    """Namespace mimicking scipy.spatial.distance."""

    @staticmethod
    def hamming(u, v):
        """Compute Hamming distance between two 1-D arrays."""
        from tinyops.ops.text.hamming_distance import hamming_distance
        return hamming_distance(u, v)


class _Spatial:
    """Namespace mimicking scipy.spatial."""
    distance = _SpatialDistance()


spatial = _Spatial()