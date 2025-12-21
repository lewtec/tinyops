import wave
import numpy as np
from tinygrad import Tensor, dtypes

def decode_wav(path: str) -> tuple[int, Tensor]:
    """
    Decodes a WAV file into a tensor.
    NOTE: This function is not implemented in pure tinygrad, as tinygrad is not suited for file I/O and byte manipulation.
    We use the standard library `wave` and `numpy` for the decoding process.
    """
    with wave.open(path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()

        frames = wf.readframes(n_frames)

    # Convert byte data to numpy array
    if sampwidth == 1:
        dtype = np.uint8
    elif sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 3:
        # 24-bit audio is not directly supported by numpy, read as 3 bytes and convert
        # Pad with a zero byte to make it 32-bit and then view as int32
        padded_frames = bytearray()
        for i in range(0, len(frames), 3):
            padded_frames.extend(frames[i:i+3] + b'\x00')
        np_arr = np.frombuffer(padded_frames, dtype=np.int32)
        # Shift right to get original 24-bit value
        np_arr = np_arr >> 8
    elif sampwidth == 4:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if sampwidth != 3:
      np_arr = np.frombuffer(frames, dtype=dtype)

    # Reshape for multi-channel audio
    if n_channels > 1:
        np_arr = np_arr.reshape(-1, n_channels)

    # Convert to float32 tensor and normalize
    if sampwidth == 1:
        tensor = Tensor(np_arr, dtype=dtypes.uint8).cast(dtypes.float32) / 128.0 - 1.0
    else:
        tensor = Tensor(np_arr, dtype=dtypes.int16 if sampwidth==2 else dtypes.int32).cast(dtypes.float32) / float(2**(sampwidth*8-1))

    return framerate, tensor
