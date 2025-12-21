import numpy as np
import torch
import torchvision.transforms.functional as F
from tinygrad import Tensor

from tinyops._core import assert_close
from tinyops.image.pad import pad


def test_pad():
    # Test case 1: Constant padding with integer
    img_tensor = Tensor(np.random.rand(3, 256, 256).astype(np.float32))
    padding = 10
    fill = 0.5
    tinyops_result = pad(img_tensor, padding, fill, "constant")

    torch_tensor = torch.from_numpy(img_tensor.numpy())
    torch_result = F.pad(torch_tensor, padding, fill, "constant")

    assert_close(tinyops_result, torch_result.numpy())

    # Test case 2: Reflect padding with tuple
    img_tensor = Tensor(np.random.rand(3, 128, 128).astype(np.float32))
    padding = (10, 20)
    tinyops_result = pad(img_tensor, padding, padding_mode="reflect")

    torch_tensor = torch.from_numpy(img_tensor.numpy())
    torch_result = F.pad(torch_tensor, padding, padding_mode="reflect")

    assert_close(tinyops_result, torch_result.numpy())

    # Test case 3: Replicate padding with tuple
    img_tensor = Tensor(np.random.rand(3, 100, 150).astype(np.float32))
    padding = (5, 10, 15, 20)
    tinyops_result = pad(img_tensor, padding, padding_mode="replicate")

    torch_tensor = torch.from_numpy(img_tensor.numpy())
    torch_result = F.pad(torch_tensor, padding, padding_mode="edge")

    assert_close(tinyops_result, torch_result.numpy())

    # Test case 4: Circular padding with tuple
    img_tensor = Tensor(np.random.rand(3, 100, 150).astype(np.float32))
    padding = (5, 10, 15, 20)
    tinyops_result = pad(img_tensor, padding, padding_mode="circular")

    # PyTorch does not support circular padding, so we will compare with a manual numpy implementation
    np_img = img_tensor.numpy()
    np_result = np.pad(
        np_img,
        ((0, 0), (10, 20), (5, 15)),
        mode="wrap",
    )

    assert_close(tinyops_result, np_result)
