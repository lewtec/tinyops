import pytest
import torch
import torchaudio
from tinygrad import Tensor

from tinyops._core import assert_close, assert_one_kernel
from tinyops.audio.mu_law_encode import mu_law_encode


def _get_input():
    data = torch.linspace(-1, 1, steps=100, dtype=torch.float32)
    return Tensor(data.numpy()).realize(), data


@pytest.mark.parametrize("quantization_channels", [256, 128, 512])
@assert_one_kernel
def test_mu_law_encode(quantization_channels):
    x, data = _get_input()

    transform = torchaudio.transforms.MuLawEncoding(quantization_channels=quantization_channels)
    expected = transform(data)

    result = mu_law_encode(x, quantization_channels=quantization_channels).realize()

    assert_close(result, expected.numpy())
