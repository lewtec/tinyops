import array
import io
import struct
import wave

from tinygrad import Tensor, dtypes

MAXIMUM_FRAME_COUNT = 500_000_000

# PCM peak divisors for normalizing integer samples to [-1, 1].
_UINT8_MIDPOINT = 128.0
_INT16_PEAK = 32768.0
_INT24_PEAK = 8388608.0
_INT32_PEAK = 2147483648.0

_NORMALIZATION_BY_SAMPLE_WIDTH = {
    1: _UINT8_MIDPOINT,
    2: _INT16_PEAK,
    3: _INT24_PEAK,
    4: _INT32_PEAK,
}


def _unpack_pcm_samples(raw_frames: bytes, sample_width: int, sample_count: int) -> list[float]:
    """Unpack PCM bytes into float32 samples in [-1, 1]."""
    if sample_width == 1:
        samples = array.array("B")
        samples.frombytes(raw_frames[:sample_count])
        return [(value - _UINT8_MIDPOINT) / _UINT8_MIDPOINT for value in samples]

    if sample_width == 2:
        samples = array.array("h")
        samples.frombytes(raw_frames[: sample_count * 2])
        return [value / _INT16_PEAK for value in samples]

    if sample_width == 3:
        floats: list[float] = []
        for index in range(sample_count):
            offset = index * 3
            sample_bytes = raw_frames[offset : offset + 3]
            sign_extension = b"\x00" if sample_bytes[2] < 128 else b"\xff"
            value = struct.unpack("<i", sample_bytes + sign_extension)[0]
            floats.append(value / _INT24_PEAK)
        return floats

    if sample_width == 4:
        samples = array.array("i")
        samples.frombytes(raw_frames[: sample_count * 4])
        return [value / _INT32_PEAK for value in samples]

    raise ValueError(f"Unsupported sample width: {sample_width}")


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
                raise ValueError(f"WAV file frame count {frame_count} exceeds limit of {MAXIMUM_FRAME_COUNT}.")

            raw_frames = wav_file.readframes(frame_count)

    expected_size = frame_count * channel_count * sample_width
    if len(raw_frames) < expected_size:
        raise ValueError(
            f"WAV data is smaller than expected. Header: {expected_size} bytes, actual: {len(raw_frames)} bytes."
        )

    if sample_width not in _NORMALIZATION_BY_SAMPLE_WIDTH:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    sample_count = frame_count * channel_count
    floats = _unpack_pcm_samples(raw_frames, sample_width, sample_count)

    if channel_count == 1:
        shaped = [[value] for value in floats]
    else:
        shaped = [floats[index : index + channel_count] for index in range(0, sample_count, channel_count)]

    return sample_rate, Tensor(shaped, dtype=dtypes.float32)
