import pytest
import torch
import torchvision.transforms.functional as F
from tinygrad import Tensor

from tinyops._core import assert_close, assert_one_kernel
from tinyops.image.pad import pad


def _get_input_color():
    img_torch = torch.randn(10, 12, 3)
    return Tensor(img_torch.numpy()).realize(), img_torch


def _get_input_grayscale():
    img_torch = torch.randn(10, 12)
    return Tensor(img_torch.numpy()).realize(), img_torch


@pytest.mark.parametrize("padding", [1, (2, 3), (1, 2, 3, 4)])
@assert_one_kernel
def test_pad_constant_color(padding):
    """Test pad with constant mode for a color image."""
    tensor_img, img_torch = _get_input_color()

    result = pad(tensor_img, padding, fill=0, padding_mode="constant").realize()
    # To test against torchvision, we need to permute from (H, W, C) to (C, H, W) and back
    expected_torch = F.pad(img_torch.permute(2, 0, 1), padding, fill=0, padding_mode="constant").permute(1, 2, 0)

    assert_close(result, expected_torch.numpy())


@pytest.mark.parametrize("padding", [1, (2, 3), (1, 2, 3, 4)])
@assert_one_kernel
def test_pad_constant_grayscale(padding):
    """Test pad with constant mode for a grayscale image."""
    tensor_img, img_torch = _get_input_grayscale()

    result = pad(tensor_img, padding, fill=0, padding_mode="constant").realize()
    expected_torch = F.pad(img_torch, padding, fill=0, padding_mode="constant")

    assert_close(result, expected_torch.numpy())


@pytest.mark.parametrize("padding", [1, (2, 3), (1, 2, 3, 4)])
@assert_one_kernel
def test_pad_reflect_color(padding):
    """Test pad with reflect mode for a color image."""
    tensor_img, img_torch = _get_input_color()

    result = pad(tensor_img, padding, padding_mode="reflect").realize()
    expected_torch = F.pad(img_torch.permute(2, 0, 1), padding, padding_mode="reflect").permute(1, 2, 0)

    assert_close(result, expected_torch.numpy())


@pytest.mark.parametrize("padding", [1, (2, 3), (1, 2, 3, 4)])
@assert_one_kernel
def test_pad_reflect_grayscale(padding):
    """Test pad with reflect mode for a grayscale image."""
    tensor_img, img_torch = _get_input_grayscale()

    result = pad(tensor_img, padding, padding_mode="reflect").realize()
    expected_torch = F.pad(img_torch, padding, padding_mode="reflect")

    assert_close(result, expected_torch.numpy())


@pytest.mark.parametrize("padding", [1, (2, 3), (1, 2, 3, 4)])
@assert_one_kernel
def test_pad_constant_4d(padding):
    """Test pad with constant mode for a 4D input (H, W, N, C)."""
    # Create 4D input: (H=10, W=12, N=2, C=3)
    img_torch = torch.randn(10, 12, 2, 3)
    tensor_img = Tensor(img_torch.numpy()).realize()

    result = pad(tensor_img, padding, fill=0, padding_mode="constant").realize()
    # To test against torchvision, we need to permute from (H, W, N, C) to (N, C, H, W) and back
    expected_torch = F.pad(img_torch.permute(2, 3, 0, 1), padding, fill=0, padding_mode="constant").permute(2, 3, 0, 1)

    assert_close(result, expected_torch.numpy())
