from tinygrad import Tensor, dtypes
import numpy as np
import io
import wave
import struct

# A reasonable limit to prevent DoS from malformed WAV headers.
# This allows for very large files (e.g., hours of multi-channel audio)
# while preventing absurdly large memory allocation requests.
MAX_WAV_FRAMES = 500_000_000  # 500 million frames
MAX_WAV_CHANNELS = 256
MAX_WAV_FRAMERATE = 1_000_000


def decode_wav(wav_bytes: bytes) -> tuple[int, Tensor]:
  """
  Decodes WAV audio bytes into a tinygrad.Tensor.

  Args:
    wav_bytes: The WAV audio data in bytes.

  Returns:
    A tuple containing the sample rate (int) and the audio data as a Tensor.
    The tensor will have a shape of (frames, channels) and will be of dtype float32,
    normalized to the range [-1.0, 1.0].
  """
  with io.BytesIO(wav_bytes) as bio:
    with wave.open(bio, 'rb') as wf:
      n_channels = wf.getnchannels()
      sampwidth = wf.getsampwidth()
      framerate = wf.getframerate()
      n_frames = wf.getnframes()

      # 🛡️ Sentinel: Add security check to prevent DoS attack.
      # A malformed WAV header with a huge n_frames value could cause
      # a massive memory allocation and crash the system.
      if n_frames > MAX_WAV_FRAMES:
        raise ValueError(f"WAV file frame count {n_frames} exceeds the security limit of {MAX_WAV_FRAMES}.")
      if n_channels > MAX_WAV_CHANNELS:
        raise ValueError(f"WAV file channel count {n_channels} exceeds the security limit of {MAX_WAV_CHANNELS}.")
      if framerate > MAX_WAV_FRAMERATE:
        raise ValueError(f"WAV file framerate {framerate} exceeds the security limit of {MAX_WAV_FRAMERATE}.")

      frames = wf.readframes(n_frames)

  # 🛡️ Sentinel: Add security check to prevent DoS from truncated data chunk.
  # If the header's frame count is correct but the data chunk is smaller,
  # the processing loop could read out of bounds, causing a crash.
  expected_data_size = n_frames * n_channels * sampwidth
  if len(frames) < expected_data_size:
    raise ValueError(
        f"WAV data chunk is smaller than expected. "
        f"Header specifies {expected_data_size} bytes, but data is {len(frames)} bytes."
    )

  # Determine numpy dtype from sample width
  if sampwidth == 1:
    dtype = np.uint8
  elif sampwidth == 2:
    dtype = np.int16
  elif sampwidth == 3:
    # 24-bit audio needs special handling
    data = np.empty((n_frames, n_channels), dtype=np.int32)
    bytes_per_sample = 3
    for i in range(n_frames * n_channels):
        offset = i * bytes_per_sample
        sample_bytes = frames[offset:offset+bytes_per_sample]
        # Pad with a sign-extending byte
        sample_bytes += b'\x00' if sample_bytes[2] < 128 else b'\xff'
        data.flat[i] = struct.unpack('<i', sample_bytes)[0]
    np_array = data
  elif sampwidth == 4:
    dtype = np.int32
  else:
    raise ValueError(f"Unsupported sample width: {sampwidth}")

  if sampwidth != 3:
    np_array = np.frombuffer(frames, dtype=dtype)

  # Reshape for multi-channel audio
  if n_channels > 1:
    np_array = np_array.reshape(-1, n_channels)
  else:
    np_array = np_array.reshape(-1, 1)


  # Normalize to float32
  if sampwidth == 1: # uint8 -> [0, 255]
    float_array = (np_array.astype(np.float32) - 128.0) / 128.0
  else:
    if sampwidth == 2: # int16 -> [-32768, 32767]
      norm_factor = 32768.0
    elif sampwidth == 3: # 24-bit stored in int32
      norm_factor = 8388608.0 # 2**23
    elif sampwidth == 4: # int32
      norm_factor = 2147483648.0 # 2**31
    float_array = np_array.astype(np.float32) / norm_factor

  return framerate, Tensor(float_array, dtype=dtypes.float32)
