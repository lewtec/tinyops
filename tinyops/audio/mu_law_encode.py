from tinygrad import Tensor, dtypes

def mu_law_encode(x: Tensor, quantization_channels: int = 256) -> Tensor:
    """
    Encode signal based on mu-law companding.
    Args:
        x (Tensor): Input tensor. Must be in range [-1, 1].
        quantization_channels (int): Number of channels.
    Returns:
        Tensor: Encoded tensor.
    """
    mu = Tensor([quantization_channels - 1], dtype=dtypes.float32)
    x_float = x.cast(dtypes.float32)

    x_mu = x_float.sign() * (1 + mu * x_float.abs()).log() / (1 + mu).log()
    encoded = ((x_mu + 1) / 2 * mu + 0.5).cast(dtypes.int64)
    return encoded
