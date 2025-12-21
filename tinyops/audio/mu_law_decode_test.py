import torch
import torchaudio
from tinygrad import Tensor
from tinyops.audio.mu_law_decode import mu_law_decode
from tinyops._core import assert_close

def test_mu_law_decode():
    data = torch.linspace(0, 255, steps=256, dtype=torch.float32)
    x = Tensor(data.numpy())

    transform = torchaudio.transforms.MuLawDecoding(quantization_channels=256)
    expected = transform(data)

    result = mu_law_decode(x, quantization_channels=256)

    assert_close(result, expected.numpy())
