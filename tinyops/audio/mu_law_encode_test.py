import torch
import torchaudio
from tinygrad import Tensor
from tinyops.audio.mu_law_encode import mu_law_encode
from tinyops._core import assert_close

def test_mu_law_encode():
    data = torch.linspace(-1, 1, steps=100, dtype=torch.float32)
    x = Tensor(data.numpy())

    transform = torchaudio.transforms.MuLawEncoding(quantization_channels=256)
    expected = transform(data)

    result = mu_law_encode(x, quantization_channels=256)

    assert_close(result, expected.numpy())
