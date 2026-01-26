import io
import struct
import wave

import numpy as np
from tinygrad import Tensor


def encode_wav(tensor: Tensor, sample_rate: int, sampwidth: int = 2) -> bytes:
    """
    Encodes a tinygrad.Tensor into WAV audio bytes.

    Args:
      tensor: The audio data as a Tensor. Must be of shape (frames, channels)
              and dtype float32, with values in the range [-1.0, 1.0].
      sample_rate: The sample rate of the audio.
      sampwidth: The sample width in bytes (1, 2, 3, or 4).

    Returns:
      The WAV audio data in bytes.
    """
    if not (tensor.dtype.name == "float32" or tensor.dtype.name == "float"):
        raise TypeError(f"Input tensor must be float32, but got {tensor.dtype}")

    float_array = tensor.numpy()
    n_frames, n_channels = float_array.shape

    if sampwidth == 1:  # uint8
        norm_factor = 128.0
        np_array = (float_array * norm_factor + 128.0).clip(0, 255).astype(np.uint8)
    else:
        if sampwidth == 2:  # int16
            norm_factor = 32767.0
            dtype = np.int16
        elif sampwidth == 3:  # 24-bit
            norm_factor = 8388607.0  # 2**23 - 1
            dtype = np.int32  # Store in int32 for manipulation
        elif sampwidth == 4:  # int32
            norm_factor = 2147483647.0  # 2**31 - 1
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")

        # Ensure the max value for signed integers is not exceeded
        max_val = 2 ** (sampwidth * 8 - 1) - 1
        np_array = (float_array * norm_factor).clip(-max_val, max_val).astype(dtype)

    with io.BytesIO() as bio:
        with wave.open(bio, "wb") as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(sample_rate)
            wf.setnframes(n_frames)

            if sampwidth == 3:
                # Manually pack 24-bit samples
                packed_data = bytearray()
                for sample in np_array.flat:
                    packed_data.extend(struct.pack("<i", int(sample))[:3])
                wf.writeframes(packed_data)
            else:
                wf.writeframes(np_array.tobytes())

        return bio.getvalue()
