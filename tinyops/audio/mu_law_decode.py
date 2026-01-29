from tinygrad import Tensor, dtypes


def mu_law_decode(x: Tensor, quantization_channels: int = 256) -> Tensor:
    """
    Decode mu-law encoded signal.

    Reverses the mu-law companding process to expand the dynamic range back to the
    original linear scale.

    Args:
        x: Input tensor with integer values in range [0, quantization_channels - 1].
        quantization_channels: Number of quantization levels used during encoding (usually 256).

    Returns:
        Decoded tensor with floating point values in range [-1, 1].
    """
    mu = Tensor([quantization_channels - 1], dtype=dtypes.float32)
    x_float = x.cast(dtypes.float32)

    x_float = x_float / mu * 2 - 1
    decoded = x_float.sign() * (1 / mu) * ((1 + mu).pow(x_float.abs()) - 1)
    return decoded
