
import numpy as np
import pytest
from tinygrad import Tensor, dtypes
from tinyops._core import assert_close
from tinyops.io.encode_wav import encode_wav
import io

# SciPy is a test-only dependency
try:
    from scipy.io import wavfile
except ImportError:
    wavfile = None

@pytest.mark.skipif(wavfile is None, reason="scipy is not installed")
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("sampwidth, dtype", [
    (1, np.uint8),
    (2, np.int16),
    (4, np.int32),
])
def test_encode_wav_scipy_comparable(channels, sampwidth, dtype):
    sample_rate = 44100
    duration = 0.1
    n_frames = int(sample_rate * duration)

    # Generate audio data as float32 tensor
    original_data = np.sin(2 * np.pi * 440 * np.arange(n_frames * channels) / sample_rate)
    original_data = original_data.reshape(n_frames, channels).astype(np.float32)
    tensor = Tensor(original_data, dtype=dtypes.float32)

    # Encode with tinyops
    wav_bytes = encode_wav(tensor, sample_rate, sampwidth=sampwidth)

    # Decode with scipy for comparison
    rate_sp, data_sp = wavfile.read(io.BytesIO(wav_bytes))

    assert rate_sp == sample_rate

    # Convert original float data to the target integer type for comparison
    if sampwidth == 1:
      expected_data = (original_data * 128.0 + 128.0).clip(0, 255).astype(dtype)
    else:
      if sampwidth == 2:
        norm_factor = 32767.0
      else:
        norm_factor = 2**(sampwidth * 8 - 1) -1
      max_val = 2**(sampwidth * 8 - 1) - 1
      expected_data = (original_data * norm_factor).clip(-max_val, max_val).astype(dtype)

    # Scipy might read single-channel data as 1D, so reshape for consistency
    if channels == 1 and len(data_sp.shape) == 1:
        data_sp = data_sp.reshape(-1, 1)

    np.testing.assert_allclose(data_sp, expected_data, atol=1)

@pytest.mark.skipif(wavfile is None, reason="scipy is not installed")
@pytest.mark.parametrize("channels", [1, 2])
def test_encode_wav_24bit(channels):
    # This test is more complex because scipy.io.wavfile doesn't support 24-bit write.
    # We will encode, then decode with our own decode_wav function to check consistency.
    from tinyops.io.decode_wav import decode_wav

    sample_rate = 44100
    duration = 0.1
    n_frames = int(sample_rate * duration)

    # Generate audio data as float32 tensor
    original_data = np.sin(2 * np.pi * 440 * np.arange(n_frames * channels) / sample_rate)
    original_data = original_data.reshape(n_frames, channels).astype(np.float32)
    tensor = Tensor(original_data, dtype=dtypes.float32)

    # Encode with tinyops
    wav_bytes = encode_wav(tensor, sample_rate, sampwidth=3)

    # Decode with our own decoder
    rate_decoded, tensor_decoded = decode_wav(wav_bytes)

    assert rate_decoded == sample_rate
    assert_close(tensor, tensor_decoded, atol=1e-5, rtol=1e-5)
