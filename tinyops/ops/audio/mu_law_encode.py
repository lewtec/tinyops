from tinygrad import Tensor, dtypes


def mu_law_encode(waveform: Tensor, quantization_channels: int = 256) -> Tensor:
    """Encode a signal using mu-law companding.

    Reduces the dynamic range of an audio signal for quantization.

    Args:
        waveform: Input tensor with values in [-1, 1].
        quantization_channels: Number of quantization levels.

    Returns:
        Encoded tensor with integer values in [0, quantization_channels - 1].
    """
    mu = Tensor([quantization_channels - 1], dtype=dtypes.float32)
    waveform_float = waveform.cast(dtypes.float32)

    compressed = waveform_float.sign() * (1 + mu * waveform_float.abs()).log() / (1 + mu).log()
    encoded = ((compressed + 1) / 2 * mu + 0.5).cast(dtypes.int64)
    return encoded
