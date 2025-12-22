import numpy as np
import pytest
from tinygrad import dtypes
from tinyops._core import assert_close
from tinyops.io.decode_wav import decode_wav
from tinyops.test_utils import assert_one_kernel
import io
import wave
import struct

# SciPy is a test-only dependency
try:
    from scipy.io import wavfile
except ImportError:
    wavfile = None

def generate_wav_bytes(sample_rate, data):
    """Generates WAV file bytes from a NumPy array."""
    bytes_io = io.BytesIO()
    wavfile.write(bytes_io, sample_rate, data)
    return bytes_io.getvalue()

def generate_24bit_wav_bytes(sample_rate, channels, frames, data_int24):
    """Generates a 24-bit WAV file manually."""
    bytes_io = io.BytesIO()
    with wave.open(bytes_io, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(3)  # 24-bit
        wf.setframerate(sample_rate)
        wf.setnframes(frames)
        packed_data = bytearray()
        for sample in data_int24.flat:
            packed_data.extend(struct.pack('<i', sample)[:3])
        wf.writeframes(packed_data)
    return bytes_io.getvalue()


@pytest.mark.skipif(wavfile is None, reason="scipy is not installed")
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("dtype, sampwidth", [
    (np.uint8, 1),
    (np.int16, 2),
    (np.int32, 4)
])
@assert_one_kernel
def test_decode_wav_scipy_comparable(channels, dtype, sampwidth):
    sample_rate = 44100
    duration = 0.1
    n_frames = int(sample_rate * duration)

    # Generate audio data
    if dtype == np.uint8:
        # Generate unsigned 8-bit data in the range [0, 255]
        data = (127.5 * (1 + np.sin(2 * np.pi * 440 * np.arange(n_frames * channels) / sample_rate))).astype(dtype)
    else:
        max_val = np.iinfo(dtype).max
        data = (np.sin(2 * np.pi * 440 * np.arange(n_frames * channels) / sample_rate) * max_val).astype(dtype)

    if channels > 1:
        data = data.reshape(n_frames, channels)

    # Create wav bytes
    wav_bytes = generate_wav_bytes(sample_rate, data)

    # Decode with tinyops
    rate_to, tensor_to = decode_wav(wav_bytes)
    tensor_to = tensor_to.realize()

    # Decode with scipy for comparison
    rate_sp, data_sp = wavfile.read(io.BytesIO(wav_bytes))

    # Normalize scipy data for comparison
    if sampwidth == 1: # uint8
        expected_data = (data_sp.astype(np.float32) - 128.0) / 128.0
    else: # int16, int32
        norm_factor = 2**(sampwidth * 8 - 1)
        expected_data = data_sp.astype(np.float32) / norm_factor

    if channels == 1:
        expected_data = expected_data.reshape(-1, 1)

    assert rate_to == rate_sp
    assert_close(tensor_to, expected_data, atol=1e-5, rtol=1e-5)

@pytest.mark.skipif(wavfile is None, reason="scipy is not installed")
@pytest.mark.parametrize("channels", [1, 2])
@assert_one_kernel
def test_decode_wav_24bit(channels):
    sample_rate = 44100
    duration = 0.1
    n_frames = int(sample_rate * duration)

    # Generate 24-bit audio data
    max_val_24bit = 2**23 - 1
    # Create data as int32, but ensure it fits within 24-bit signed range
    data_int32 = (np.sin(2 * np.pi * 440 * np.arange(n_frames * channels) / sample_rate) * max_val_24bit).astype(np.int32)
    if channels > 1:
        data_int32 = data_int32.reshape(n_frames, channels)

    # Create 24-bit wav bytes manually
    wav_bytes = generate_24bit_wav_bytes(sample_rate, channels, n_frames, data_int32)

    # Decode with tinyops
    rate_to, tensor_to = decode_wav(wav_bytes)
    tensor_to = tensor_to.realize()

    # Expected data after normalization
    norm_factor = 2**23
    expected_data = data_int32.astype(np.float32) / norm_factor
    if channels == 1:
        expected_data = expected_data.reshape(-1, 1)

    assert rate_to == sample_rate
    assert_close(tensor_to, expected_data, atol=1e-5, rtol=1e-5)
