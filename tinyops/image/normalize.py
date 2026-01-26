from tinygrad import Tensor, dtypes

# Normalization types from OpenCV
NORM_MINMAX = 32


def normalize(src: Tensor, alpha: float = 0, beta: float = 255, norm_type: int = NORM_MINMAX) -> Tensor:
    """
    Normalizes the range of a tensor.
    This implementation is designed to be compatible with OpenCV's normalize function.
    """
    if norm_type == NORM_MINMAX:
        src_min = src.min()
        src_max = src.max()

        den = src_max - src_min
        is_const = den.eq(0)

        # Avoid division by zero by replacing den with 1.0 where it's 0.
        # The result of this branch of the calculation will be discarded by the final `where`.
        safe_den = is_const.where(1.0, den)
        scale = (beta - alpha) / safe_den
        normalized = (src - src_min) * scale + alpha

        # Where the image was constant, return a tensor filled with alpha. Otherwise, return the normalized result.
        result = is_const.where(Tensor.full(src.shape, alpha, dtype=src.dtype), normalized)

        if src.dtype in [dtypes.uint8, dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64]:
            return result.cast(src.dtype)
        return result

    raise NotImplementedError(f"Normalization type {norm_type} not implemented yet.")
