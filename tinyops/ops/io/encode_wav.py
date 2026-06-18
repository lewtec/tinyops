import io
import struct
import wave

import numpy as np
from tinygrad import Tensor


def encode_wav(audio: Tensor, sample_rate: int, sample_width: int = 2) -> bytes:
    """Encode a tensor as WAV audio bytes.

    Args:
        audio: Audio tensor of shape (frames, channels), dtype float32,
            values in [-1.0, 1.0].
        sample_rate: Audio sample rate in Hz.
        sample_width: Bytes per sample (1, 2, 3, or 4).

    Returns:
        WAV file as bytes.

    Raises:
        TypeError: If input is not float32.
        ValueError: If sample_width is not supported.
    """
    if not (audio.dtype.name == "float32" or audio.dtype.name == "float"):
        raise TypeError(f"Input tensor must be float32, but got {audio.dtype}")

    float_array = audio.numpy()
    frame_count, channel_count = float_array.shape

    if sample_width == 1:
        numpy_array = (float_array * 128.0 + 128.0).clip(0, 255).astype(np.uint8)
    elif sample_width in (2, 3, 4):
        normalization_factors = {2: 32767.0, 3: 8388607.0, 4: 2147483647.0}
        factor = normalization_factors[sample_width]
        numpy_dtype = np.int16 if sample_width == 2 else np.int32
        max_value = 2 ** (sample_width * 8 - 1) - 1
        numpy_array = (float_array * factor).clip(-max_value, max_value).astype(numpy_dtype)
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(channel_count)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.setnframes(frame_count)

            if sample_width == 3:
                packed = bytearray()
                for sample in numpy_array.flat:
                    packed.extend(struct.pack("<i", int(sample))[:3])
                wav_file.writeframes(packed)
            else:
                wav_file.writeframes(numpy_array.tobytes())

        return buffer.getvalue()
