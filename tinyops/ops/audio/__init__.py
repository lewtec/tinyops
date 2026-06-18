"""Audio processing operations: encoding, transforms, masking."""

from .amplitude_to_decibels import SpectrogramScale, amplitude_to_decibels
from .fade import FadeShape, fade
from .frequency_mask import frequency_mask
from .mu_law_decode import mu_law_decode
from .mu_law_encode import mu_law_encode
from .time_mask import time_mask
