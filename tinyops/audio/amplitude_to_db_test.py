import torch
import torchaudio
from tinygrad import Tensor
from tinyops.audio.amplitude_to_db import amplitude_to_db
from tinyops._core import assert_close

def test_amplitude_to_db_magnitude():
    sample_rate = 16000
    waveform = torch.randn(1, sample_rate * 4)

    spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=1024, power=1)
    spectrogram = spectrogram_transform(waveform)

    tinygrad_db = amplitude_to_db(Tensor(spectrogram.numpy()), stype="magnitude", top_db=80)
    torch_db = torchaudio.transforms.AmplitudeToDB(stype="magnitude", top_db=80)(spectrogram)

    assert_close(tinygrad_db, torch_db.numpy(), atol=1e-5, rtol=1e-5)

def test_amplitude_to_db_power():
    sample_rate = 16000
    waveform = torch.randn(1, sample_rate * 4)

    spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=1024, power=2)
    spectrogram = spectrogram_transform(waveform)

    tinygrad_db = amplitude_to_db(Tensor(spectrogram.numpy()), stype="power", top_db=80)
    torch_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)(spectrogram)

    assert_close(tinygrad_db, torch_db.numpy(), atol=1e-5, rtol=1e-5)
