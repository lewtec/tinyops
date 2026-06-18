"""Audio processing operations: encoding, transforms, masking."""

from .mu_law_encode import mu_law_encode
from .mu_law_decode import mu_law_decode
from .amplitude_to_decibels import amplitude_to_decibels, SpectrogramScale
from .fade import fade, FadeShape
from .frequency_mask import frequency_mask
from .time_mask import time_mask
