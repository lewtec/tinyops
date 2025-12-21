import numpy as np
import torch
import torchvision.transforms.functional as F
from tinygrad import Tensor

from tinyops._core import assert_close
from tinyops.image.center_crop import center_crop


def test_center_crop():
    # Test case 1: Square crop
    img_tensor = Tensor(np.random.rand(3, 256, 256).astype(np.float32))
    output_size = 128
    tinyops_result = center_crop(img_tensor, output_size)

    torch_tensor = torch.from_numpy(img_tensor.numpy())
    torch_result = F.center_crop(torch_tensor, output_size)

    assert_close(tinyops_result, torch_result.numpy())

    # Test case 2: Rectangular crop
    img_tensor = Tensor(np.random.rand(3, 256, 300).astype(np.float32))
    output_size = (128, 200)
    tinyops_result = center_crop(img_tensor, output_size)

    torch_tensor = torch.from_numpy(img_tensor.numpy())
    torch_result = F.center_crop(torch_tensor, output_size)

    assert_close(tinyops_result, torch_result.numpy())
