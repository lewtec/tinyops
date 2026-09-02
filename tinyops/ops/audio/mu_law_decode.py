from tinygrad import Tensor, dtypes


def mu_law_decode(encoded: Tensor, quantization_channels: int = 256) -> Tensor:
    """Decode a mu-law encoded signal back to linear scale.

    Args:
        encoded: Input tensor with integer values in [0, quantization_channels - 1].
        quantization_channels: Number of quantization levels used during encoding.

    Returns:
        Decoded tensor with values in [-1, 1].
    """
    mu = Tensor([quantization_channels - 1], dtype=dtypes.float32)
    encoded_float = encoded.cast(dtypes.float32)

    normalized = encoded_float / mu * 2 - 1
    decoded = normalized.sign() * (1 / mu) * ((1 + mu).pow(normalized.abs()) - 1)
    return decoded
