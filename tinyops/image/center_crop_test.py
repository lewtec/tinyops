import torch
import torchvision.transforms as T
from tinygrad import Tensor
from tinyops._core import assert_close
from tinyops.image.center_crop import center_crop

def test_center_crop_smaller():
    img_tensor = Tensor.rand(3, 256, 256)
    size = 128

    tinyops_result = center_crop(img_tensor, size)

    torch_transform = T.CenterCrop(size)
    torch_result = torch_transform(torch.from_numpy(img_tensor.numpy()))

    assert_close(tinyops_result, torch_result.numpy())

def test_center_crop_larger():
    img_tensor = Tensor.rand(3, 128, 128)
    size = 256

    tinyops_result = center_crop(img_tensor, size)

    torch_transform = T.CenterCrop(size)
    torch_result = torch_transform(torch.from_numpy(img_tensor.numpy()))

    assert_close(tinyops_result, torch_result.numpy())

def test_center_crop_tuple():
    img_tensor = Tensor.rand(3, 256, 256)
    size = (100, 150)

    tinyops_result = center_crop(img_tensor, size)

    torch_transform = T.CenterCrop(size)
    torch_result = torch_transform(torch.from_numpy(img_tensor.numpy()))

    assert_close(tinyops_result, torch_result.numpy())

def test_center_crop_batched():
    img_tensor = Tensor.rand(4, 3, 256, 256)
    size = 128

    tinyops_result = center_crop(img_tensor, size)

    # Torch CenterCrop doesn't directly support batching in the same way,
    # so we apply it to each image in the batch and stack the results.
    torch_transform = T.CenterCrop(size)
    torch_result = torch.stack([torch_transform(torch.from_numpy(img.numpy())) for img in img_tensor])

    assert_close(tinyops_result, torch_result.numpy())
