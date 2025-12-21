import torch
import torchvision.transforms.functional as F
import pytest
from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.image.pad import pad

@pytest.mark.parametrize("padding", [1, (2, 3), (1, 2, 3, 4)])
def test_pad_constant_color(padding):
    """Test pad with constant mode for a color image."""
    img_torch = torch.randn(10, 12, 3)
    tensor_img = Tensor(img_torch.numpy())

    result = pad(tensor_img, padding, fill=0, padding_mode="constant")
    # To test against torchvision, we need to permute from (H, W, C) to (C, H, W) and back
    expected_torch = F.pad(img_torch.permute(2, 0, 1), padding, fill=0, padding_mode="constant").permute(1, 2, 0)

    assert_close(result, expected_torch.numpy())

@pytest.mark.parametrize("padding", [1, (2, 3), (1, 2, 3, 4)])
def test_pad_constant_grayscale(padding):
    """Test pad with constant mode for a grayscale image."""
    img_torch = torch.randn(10, 12)
    tensor_img = Tensor(img_torch.numpy())

    result = pad(tensor_img, padding, fill=0, padding_mode="constant")
    expected_torch = F.pad(img_torch, padding, fill=0, padding_mode="constant")

    assert_close(result, expected_torch.numpy())
