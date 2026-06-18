"""torchaudio compatibility layer.

Provides torchaudio.transforms-compatible class signatures that delegate to tinyops.ops.
"""

from tinygrad import Tensor

from tinyops.ops.audio.mu_law_encode import mu_law_encode as _mu_law_encode
from tinyops.ops.audio.mu_law_decode import mu_law_decode as _mu_law_decode
from tinyops.ops.audio.amplitude_to_decibels import amplitude_to_decibels as _amp_to_db, SpectrogramScale
from tinyops.ops.audio.fade import fade as _fade, FadeShape
from tinyops.ops.audio.frequency_mask import frequency_mask as _freq_mask
from tinyops.ops.audio.time_mask import time_mask as _time_mask


class _Transforms:
    """Namespace mimicking torchaudio.transforms."""

    class MuLawEncoding:
        """Encode signal using mu-law companding."""

        def __init__(self, quantization_channels: int = 256):
            self.quantization_channels = quantization_channels

        def __call__(self, waveform: Tensor) -> Tensor:
            return _mu_law_encode(waveform, quantization_channels=self.quantization_channels)

    class MuLawDecoding:
        """Decode mu-law encoded signal."""

        def __init__(self, quantization_channels: int = 256):
            self.quantization_channels = quantization_channels

        def __call__(self, encoded: Tensor) -> Tensor:
            return _mu_law_decode(encoded, quantization_channels=self.quantization_channels)

    class AmplitudeToDB:
        """Convert spectrogram to decibel scale."""

        _SCALE_MAP = {
            "power": SpectrogramScale.POWER,
            "magnitude": SpectrogramScale.MAGNITUDE,
        }

        def __init__(self, stype: str = "power", top_db: float | None = 80.0):
            self.stype = stype
            self.top_db = top_db

        def __call__(self, spectrogram: Tensor) -> Tensor:
            scale = self._SCALE_MAP[self.stype]
            return _amp_to_db(spectrogram, scale=scale, dynamic_range=self.top_db)

    class Fade:
        """Apply fade in and/or fade out to a waveform."""

        _SHAPE_MAP = {
            "linear": FadeShape.LINEAR,
            "exponential": FadeShape.EXPONENTIAL,
            "logarithmic": FadeShape.LOGARITHMIC,
            "quarter_sine": FadeShape.QUARTER_SINE,
            "half_sine": FadeShape.HALF_SINE,
        }

        def __init__(self, fade_in_len: int = 0, fade_out_len: int = 0, fade_shape: str = "linear"):
            self.fade_in_len = fade_in_len
            self.fade_out_len = fade_out_len
            self.fade_shape = fade_shape

        def __call__(self, waveform: Tensor) -> Tensor:
            shape = self._SHAPE_MAP[self.fade_shape]
            return _fade(waveform, fade_in_length=self.fade_in_len, fade_out_length=self.fade_out_len, shape=shape)

    class FrequencyMasking:
        """Apply masking to a frequency dimension of a spectrogram."""

        def __init__(self, freq_mask_param: int):
            self.freq_mask_param = freq_mask_param

        def __call__(self, spectrogram: Tensor) -> Tensor:
            return _freq_mask(spectrogram, maximum_mask_length=self.freq_mask_param)

    class TimeMasking:
        """Apply masking to the time dimension of a spectrogram."""

        def __init__(self, time_mask_param: int):
            self.time_mask_param = time_mask_param

        def __call__(self, spectrogram: Tensor) -> Tensor:
            return _time_mask(spectrogram, maximum_mask_length=self.time_mask_param)


transforms = _Transforms()