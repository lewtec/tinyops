import pytest
import torch
import torchvision.transforms as T
from tinygrad import Tensor

from tinyops._core import assert_close, assert_one_kernel
from tinyops.image.center_crop import center_crop


def _get_input(shape):
    return Tensor.rand(*shape).realize()


@pytest.mark.parametrize("size", [128])
@assert_one_kernel
def test_center_crop_smaller(size):
    img_tensor = _get_input((3, 256, 256))

    tinyops_result = center_crop(img_tensor, size).realize()

    torch_transform = T.CenterCrop(size)
    torch_result = torch_transform(torch.from_numpy(img_tensor.numpy()))

    assert_close(tinyops_result, torch_result.numpy())


@pytest.mark.parametrize("size", [256])
@assert_one_kernel
def test_center_crop_larger(size):
    img_tensor = _get_input((3, 128, 128))

    tinyops_result = center_crop(img_tensor, size).realize()

    torch_transform = T.CenterCrop(size)
    torch_result = torch_transform(torch.from_numpy(img_tensor.numpy()))

    assert_close(tinyops_result, torch_result.numpy())


@pytest.mark.parametrize("size", [(100, 150)])
@assert_one_kernel
def test_center_crop_tuple(size):
    img_tensor = _get_input((3, 256, 256))

    tinyops_result = center_crop(img_tensor, size).realize()

    torch_transform = T.CenterCrop(size)
    torch_result = torch_transform(torch.from_numpy(img_tensor.numpy()))

    assert_close(tinyops_result, torch_result.numpy())


@pytest.mark.parametrize("size", [128])
@assert_one_kernel
def test_center_crop_batched(size):
    img_tensor = _get_input((4, 3, 256, 256))

    tinyops_result = center_crop(img_tensor, size).realize()

    # Torch CenterCrop doesn't directly support batching in the same way,
    # so we apply it to each image in the batch and stack the results.
    torch_transform = T.CenterCrop(size)
    torch_result = torch.stack([torch_transform(torch.from_numpy(img.numpy())) for img in img_tensor])

    assert_close(tinyops_result, torch_result.numpy())
