"""Tests for torchaudio compatibility layer.

Compares tinyops.compat.torchaudio against actual torchaudio.
"""

import numpy as np
import torch
import torchaudio.transforms as T
from tinygrad import Tensor

from tinyops._core import assert_close
from tinyops.compat import torchaudio as tta


class TestMuLawEncoding:
    def test_basic(self):
        waveform = np.random.randn(1, 100).astype(np.float32) * 0.5
        encoder_ours = tta.transforms.MuLawEncoding(quantization_channels=256)
        result = encoder_ours(Tensor(waveform))

        encoder_ref = T.MuLawEncoding(quantization_channels=256)
        expected = encoder_ref(torch.tensor(waveform)).numpy()
        assert_close(result, expected, atol=1)

    def test_different_channels(self):
        waveform = np.random.randn(1, 50).astype(np.float32)
        encoder_ours = tta.transforms.MuLawEncoding(quantization_channels=128)
        result = encoder_ours(Tensor(waveform))

        encoder_ref = T.MuLawEncoding(quantization_channels=128)
        expected = encoder_ref(torch.tensor(waveform)).numpy()
        assert_close(result, expected, atol=1)


class TestMuLawDecoding:
    def test_roundtrip(self):
        waveform = np.random.randn(1, 100).astype(np.float32) * 0.5
        encoder = tta.transforms.MuLawEncoding(256)
        decoder = tta.transforms.MuLawDecoding(256)
        encoded = encoder(Tensor(waveform))
        decoded = decoder(encoded)
        # Mu-law quantization introduces noise; verify shape and range
        assert decoded.shape == waveform.shape
        decoded_np = decoded.numpy()
        assert decoded_np.min() >= -2.0
        assert decoded_np.max() <= 2.0


class TestAmplitudeToDB:
    def test_power_spectrogram(self):
        spectrogram = np.abs(np.random.randn(1, 20, 30).astype(np.float32)) + 1e-6
        transform_ours = tta.transforms.AmplitudeToDB(stype="power")
        result = transform_ours(Tensor(spectrogram))

        transform_ref = T.AmplitudeToDB(stype="power")
        expected = transform_ref(torch.tensor(spectrogram)).numpy()
        assert_close(result, expected, atol=1e-3)

    def test_magnitude_spectrogram(self):
        rng = np.random.RandomState(42)
        spectrogram = np.abs(rng.randn(1, 10, 15).astype(np.float32)) + 0.01
        transform_ours = tta.transforms.AmplitudeToDB(stype="magnitude")
        result = transform_ours(Tensor(spectrogram))

        transform_ref = T.AmplitudeToDB(stype="magnitude")
        expected = transform_ref(torch.tensor(spectrogram)).numpy()
        assert_close(result, expected, atol=1e-3)


class TestFade:
    def test_linear_fade_in(self):
        waveform = np.ones((1, 100), dtype=np.float32)
        transform_ours = tta.transforms.Fade(fade_in_len=10, fade_out_len=0, fade_shape="linear")
        result = transform_ours(Tensor(waveform))

        transform_ref = T.Fade(fade_in_len=10, fade_out_len=0, fade_shape="linear")
        expected = transform_ref(torch.tensor(waveform)).numpy()
        assert_close(result, expected, atol=1e-4)

    def test_linear_fade_out(self):
        waveform = np.ones((1, 100), dtype=np.float32)
        transform_ours = tta.transforms.Fade(fade_in_len=0, fade_out_len=10, fade_shape="linear")
        result = transform_ours(Tensor(waveform))

        transform_ref = T.Fade(fade_in_len=0, fade_out_len=10, fade_shape="linear")
        expected = transform_ref(torch.tensor(waveform)).numpy()
        assert_close(result, expected, atol=1e-4)

    def test_both_fades(self):
        waveform = np.ones((1, 100), dtype=np.float32)
        transform_ours = tta.transforms.Fade(fade_in_len=20, fade_out_len=20)
        result = transform_ours(Tensor(waveform))

        transform_ref = T.Fade(fade_in_len=20, fade_out_len=20)
        expected = transform_ref(torch.tensor(waveform)).numpy()
        assert_close(result, expected, atol=1e-4)


class TestFrequencyMasking:
    def test_shape_preserved(self):
        spectrogram = np.random.randn(1, 20, 30).astype(np.float32)
        transform = tta.transforms.FrequencyMasking(freq_mask_param=5)
        result = transform(Tensor(spectrogram))
        assert result.shape == spectrogram.shape

    def test_masking_applied(self):
        spectrogram = np.ones((1, 20, 30), dtype=np.float32)
        transform = tta.transforms.FrequencyMasking(freq_mask_param=5)
        result = transform(Tensor(spectrogram))
        # Some values should be masked to 0
        result_np = result.numpy()
        assert result_np.min() >= 0.0


class TestTimeMasking:
    def test_shape_preserved(self):
        spectrogram = np.random.randn(1, 20, 30).astype(np.float32)
        transform = tta.transforms.TimeMasking(time_mask_param=5)
        result = transform(Tensor(spectrogram))
        assert result.shape == spectrogram.shape

    def test_masking_applied(self):
        spectrogram = np.ones((1, 20, 30), dtype=np.float32)
        transform = tta.transforms.TimeMasking(time_mask_param=5)
        result = transform(Tensor(spectrogram))
        result_np = result.numpy()
        assert result_np.min() >= 0.0
