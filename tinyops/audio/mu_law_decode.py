from tinygrad import Tensor, dtypes


def mu_law_decode(x: Tensor, quantization_channels: int = 256) -> Tensor:
    """
    Decode mu-law encoded signal.
    Args:
        x (Tensor): Input tensor.
        quantization_channels (int): Number of channels.
    Returns:
        Tensor: Decoded tensor.
    """
    mu = Tensor([quantization_channels - 1], dtype=dtypes.float32)
    x_float = x.cast(dtypes.float32)

    x_float = x_float / mu * 2 - 1
    decoded = x_float.sign() * (1 / mu) * ((1 + mu).pow(x_float.abs()) - 1)
    return decoded
