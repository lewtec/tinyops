import os
import pytest
import numpy as np
import wave
from scipy.io.wavfile import write as write_wav
from tinyops.io.decode_wav import decode_wav
from tinyops._core import assert_close

@pytest.fixture
def mono_wav_file_16bit(tmp_path):
    samplerate = 44100
    freq = 440
    duration = 1
    t = np.linspace(0., duration, samplerate * duration, endpoint=False)
    amplitude = np.iinfo(np.int16).max * 0.5
    data = amplitude * np.sin(2. * np.pi * freq * t)

    filepath = tmp_path / "mono_16.wav"
    write_wav(filepath, samplerate, data.astype(np.int16))
    return filepath, samplerate, data / (2**15)

@pytest.fixture
def stereo_wav_file_16bit(tmp_path):
    samplerate = 44100
    freq1 = 440
    freq2 = 880
    duration = 1
    t = np.linspace(0., duration, samplerate * duration, endpoint=False)
    amplitude = np.iinfo(np.int16).max * 0.5

    data1 = amplitude * np.sin(2. * np.pi * freq1 * t)
    data2 = amplitude * np.sin(2. * np.pi * freq2 * t)

    data = np.stack([data1, data2], axis=1)

    filepath = tmp_path / "stereo_16.wav"
    write_wav(filepath, samplerate, data.astype(np.int16))
    return filepath, samplerate, data / (2**15)

@pytest.fixture
def mono_wav_file_8bit(tmp_path):
    samplerate = 44100
    freq = 440
    duration = 1
    t = np.linspace(0., duration, samplerate * duration, endpoint=False)
    data = 127.5 * (1 + np.sin(2. * np.pi * freq * t))

    filepath = tmp_path / "mono_8.wav"
    write_wav(filepath, samplerate, data.astype(np.uint8))
    expected_data = (data / 128.0) - 1.0
    return filepath, samplerate, expected_data

@pytest.fixture
def mono_wav_file_32bit(tmp_path):
    samplerate = 44100
    freq = 440
    duration = 1
    t = np.linspace(0., duration, samplerate * duration, endpoint=False)
    amplitude = np.iinfo(np.int32).max * 0.5
    data = amplitude * np.sin(2. * np.pi * freq * t)

    filepath = tmp_path / "mono_32.wav"
    write_wav(filepath, samplerate, data.astype(np.int32))
    return filepath, samplerate, data / (2**31)

def test_decode_mono_wav_16bit(mono_wav_file_16bit):
    filepath, expected_rate, expected_data = mono_wav_file_16bit
    rate, tensor = decode_wav(str(filepath))
    assert rate == expected_rate
    assert_close(tensor, expected_data, atol=1e-4)

def test_decode_stereo_wav_16bit(stereo_wav_file_16bit):
    filepath, expected_rate, expected_data = stereo_wav_file_16bit
    rate, tensor = decode_wav(str(filepath))
    assert rate == expected_rate
    assert_close(tensor, expected_data, atol=1e-4)

def test_decode_mono_wav_8bit(mono_wav_file_8bit):
    filepath, expected_rate, expected_data = mono_wav_file_8bit
    rate, tensor = decode_wav(str(filepath))
    assert rate == expected_rate
    assert_close(tensor, expected_data, atol=1e-2)

def test_decode_mono_wav_32bit(mono_wav_file_32bit):
    filepath, expected_rate, expected_data = mono_wav_file_32bit
    rate, tensor = decode_wav(str(filepath))
    assert rate == expected_rate
    assert_close(tensor, expected_data, atol=1e-4)
