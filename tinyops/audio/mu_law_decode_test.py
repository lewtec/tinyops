import torch
import torchaudio
import pytest
from tinygrad import Tensor
from tinyops.audio.mu_law_decode import mu_law_decode
from tinyops._core import assert_close
from tinyops.test_utils import assert_one_kernel

def _get_input():
    data = torch.linspace(0, 255, steps=256, dtype=torch.float32)
    return Tensor(data.numpy()).realize(), data

@pytest.mark.parametrize("quantization_channels", [256, 128, 512])
@assert_one_kernel
def test_mu_law_decode(quantization_channels):
    x, data = _get_input()

    transform = torchaudio.transforms.MuLawDecoding(quantization_channels=quantization_channels)
    expected = transform(data)

    result = mu_law_decode(x, quantization_channels=quantization_channels).realize()

    assert_close(result, expected.numpy())
