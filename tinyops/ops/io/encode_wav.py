import array
import io
import struct
import wave
from collections.abc import Sequence

from tinygrad import Tensor

_INT16_CLIP_PEAK = 32767.0
_INT24_CLIP_PEAK = 8388607.0
_INT32_CLIP_PEAK = 2147483647.0

_CLIP_PEAK_BY_SAMPLE_WIDTH = {
    2: _INT16_CLIP_PEAK,
    3: _INT24_CLIP_PEAK,
    4: _INT32_CLIP_PEAK,
}


def _flatten_audio_rows(rows: Sequence) -> list[float]:
    """Flatten a tolist() result that may be nested rows or a scalar."""
    if not rows:
        return []
    first = rows[0]
    if isinstance(first, (list, tuple)):
        flat: list[float] = []
        for row in rows:
            flat.extend(float(value) for value in row)
        return flat
    return [float(value) for value in rows]


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

    rows = audio.tolist()
    if audio.ndim == 1:
        frame_count = audio.shape[0]
        channel_count = 1
        floats = [float(value) for value in rows] if isinstance(rows, list) else [float(rows)]
    else:
        frame_count, channel_count = audio.shape
        floats = _flatten_audio_rows(rows)

    if sample_width == 1:
        pcm = array.array("B", (max(0, min(255, int(value * 128.0 + 128.0))) for value in floats))
        frame_bytes = pcm.tobytes()
    elif sample_width in _CLIP_PEAK_BY_SAMPLE_WIDTH:
        peak = _CLIP_PEAK_BY_SAMPLE_WIDTH[sample_width]
        max_value = int(peak)
        clipped = [max(-max_value, min(max_value, int(value * peak))) for value in floats]
        if sample_width == 2:
            frame_bytes = array.array("h", clipped).tobytes()
        elif sample_width == 3:
            packed = bytearray()
            for sample in clipped:
                packed.extend(struct.pack("<i", sample)[:3])
            frame_bytes = bytes(packed)
        else:
            frame_bytes = array.array("i", clipped).tobytes()
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(channel_count)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.setnframes(frame_count)
            wav_file.writeframes(frame_bytes)
        return buffer.getvalue()
