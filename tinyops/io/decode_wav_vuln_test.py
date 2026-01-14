
import pytest
import wave
import io
import struct
from unittest.mock import MagicMock
from tinyops.io.decode_wav import decode_wav, MAX_WAV_CHANNELS, MAX_WAV_FRAMERATE

def create_wav_with_custom_header(n_channels: int, framerate: int, sampwidth: int = 2, n_frames: int = 1) -> bytes:
    """
    Creates a WAV file in memory with a valid structure, then manually overwrites
    header fields with potentially invalid values. This is necessary because the
    `wave` module's reader might validate some fields (like non-zero channels)
    before our own validation code is reached.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sampwidth)
        wf.setframerate(44100)
        wf.writeframes(b'\x00' * sampwidth * n_frames)

    wav_data = bytearray(buffer.getvalue())

    struct.pack_into('<H', wav_data, 22, n_channels)
    struct.pack_into('<I', wav_data, 24, framerate)

    return bytes(wav_data)

def test_decode_wav_excessive_channels():
    """Verify our validation rejects a WAV with an excessive channel count."""
    malicious_wav = create_wav_with_custom_header(n_channels=MAX_WAV_CHANNELS + 1, framerate=44100)
    with pytest.raises(ValueError, match="exceeds the security limit"):
        decode_wav(malicious_wav)

def test_decode_wav_excessive_framerate():
    """Verify our validation rejects a WAV with an excessive framerate."""
    malicious_wav = create_wav_with_custom_header(n_channels=1, framerate=MAX_WAV_FRAMERATE + 1)
    with pytest.raises(ValueError, match="exceeds the security limit"):
        decode_wav(malicious_wav)

def test_decode_wav_zero_channels(monkeypatch):
    """
    Verify our validation rejects zero channels, using a mock to bypass the
    wave library's own validation which would otherwise raise a `wave.Error`.
    """
    mock_wf = MagicMock()
    mock_wf.getnchannels.return_value = 0
    mock_wf.getsampwidth.return_value = 2
    mock_wf.getframerate.return_value = 44100
    mock_wf.getnframes.return_value = 1
    mock_wf.readframes.return_value = b'\x00\x00'

    mock_context_manager = MagicMock()
    mock_context_manager.__enter__.return_value = mock_wf
    mock_context_manager.__exit__.return_value = None

    monkeypatch.setattr("wave.open", lambda *args, **kwargs: mock_context_manager)

    with pytest.raises(ValueError, match="invalid or exceeds the security limit"):
        decode_wav(b'dummy_wav_data')

def test_decode_wav_zero_framerate(monkeypatch):
    """
    Verify our validation rejects a zero framerate, using a mock to bypass
    the wave library's own validation.
    """
    mock_wf = MagicMock()
    mock_wf.getnchannels.return_value = 1
    mock_wf.getsampwidth.return_value = 2
    mock_wf.getframerate.return_value = 0
    mock_wf.getnframes.return_value = 1
    mock_wf.readframes.return_value = b'\x00\x00'

    mock_context_manager = MagicMock()
    mock_context_manager.__enter__.return_value = mock_wf
    mock_context_manager.__exit__.return_value = None

    monkeypatch.setattr("wave.open", lambda *args, **kwargs: mock_context_manager)

    with pytest.raises(ValueError, match="invalid or exceeds the security limit"):
        decode_wav(b'dummy_wav_data')
