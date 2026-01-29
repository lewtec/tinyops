from tinygrad import Tensor, dtypes


def mu_law_encode(x: Tensor, quantization_channels: int = 256) -> Tensor:
    """
    Encode signal based on mu-law companding.

    Mu-law encoding reduces the dynamic range of an audio signal, allowing for
    quantization with fewer bits while maintaining fidelity for lower amplitudes.
    This is commonly used in telephony and generative audio models like WaveNet.

    Args:
        x: Input tensor representing the audio signal. Values should be in range [-1, 1].
        quantization_channels: Number of quantization levels (usually 256).

    Returns:
        Encoded tensor with integer values in range [0, quantization_channels - 1].
    """
    mu = Tensor([quantization_channels - 1], dtype=dtypes.float32)
    x_float = x.cast(dtypes.float32)

    x_mu = x_float.sign() * (1 + mu * x_float.abs()).log() / (1 + mu).log()
    encoded = ((x_mu + 1) / 2 * mu + 0.5).cast(dtypes.int64)
    return encoded
