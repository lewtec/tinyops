import io
import struct
import wave

import numpy as np
from tinygrad import Tensor, dtypes

MAXIMUM_FRAME_COUNT = 500_000_000


def decode_wav(wav_bytes: bytes) -> tuple[int, Tensor]:
    """Decode WAV audio bytes into a tensor.

    Supports 8-bit unsigned, 16-bit, 24-bit, and 32-bit signed PCM.
    Output is always float32 in [-1.0, 1.0].

    Args:
        wav_bytes: Raw WAV file bytes.

    Returns:
        Tuple of (sample_rate, audio_tensor) where audio_tensor has
        shape (frames, channels).

    Raises:
        ValueError: For malformed or excessively large WAV files.
    """
    with io.BytesIO(wav_bytes) as buffer:
        with wave.open(buffer, "rb") as wav_file:
            channel_count = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()

            if frame_count > MAXIMUM_FRAME_COUNT:
                raise ValueError(
                    f"WAV file frame count {frame_count} exceeds limit of {MAXIMUM_FRAME_COUNT}."
                )

            raw_frames = wav_file.readframes(frame_count)

    expected_size = frame_count * channel_count * sample_width
    if len(raw_frames) < expected_size:
        raise ValueError(
            f"WAV data is smaller than expected. "
            f"Header: {expected_size} bytes, actual: {len(raw_frames)} bytes."
        )

    if sample_width == 1:
        numpy_dtype = np.uint8
    elif sample_width == 2:
        numpy_dtype = np.int16
    elif sample_width == 3:
        data = np.empty((frame_count, channel_count), dtype=np.int32)
        for i in range(frame_count * channel_count):
            offset = i * 3
            sample_bytes = raw_frames[offset : offset + 3]
            sample_bytes += b"\x00" if sample_bytes[2] < 128 else b"\xff"
            data.flat[i] = struct.unpack("<i", sample_bytes)[0]
        numpy_array = data
    elif sample_width == 4:
        numpy_dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    if sample_width != 3:
        numpy_array = np.frombuffer(raw_frames, dtype=numpy_dtype)

    numpy_array = numpy_array.reshape(-1, channel_count) if channel_count > 1 else numpy_array.reshape(-1, 1)

    normalization_factors = {1: 128.0, 2: 32768.0, 3: 8388608.0, 4: 2147483648.0}
    factor = normalization_factors[sample_width]

    if sample_width == 1:
        float_array = (numpy_array.astype(np.float32) - 128.0) / 128.0
    else:
        float_array = numpy_array.astype(np.float32) / factor

    return sample_rate, Tensor(float_array, dtype=dtypes.float32)
