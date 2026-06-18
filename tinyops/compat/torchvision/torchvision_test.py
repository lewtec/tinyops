"""Tests for torchvision compatibility layer.

Compares tinyops.compat.torchvision against actual torchvision.
"""

import numpy as np
from tinygrad import Tensor

from tinyops._core import assert_close
from tinyops.compat import torchvision as ttv


class TestCenterCrop:
    def test_square_crop(self):
        img = np.random.randn(100, 100).astype(np.float32)
        crop = ttv.transforms.CenterCrop(50)
        result = crop(Tensor(img))
        assert result.shape == (50, 50)

    def test_rectangular_crop(self):
        img = np.random.randn(100, 80).astype(np.float32)
        crop = ttv.transforms.CenterCrop((40, 30))
        result = crop(Tensor(img))
        assert result.shape == (40, 30)

    def test_center_values(self):
        img = np.zeros((10, 10), dtype=np.float32)
        img[3:7, 3:7] = 1.0  # center region
        crop = ttv.transforms.CenterCrop(4)
        result = crop(Tensor(img))
        assert_close(result, np.ones((4, 4), dtype=np.float32))

    def test_3d_image(self):
        # center_crop operates on (*, H, W), e.g. (C, H, W)
        img = np.random.randn(3, 100, 100).astype(np.float32)
        crop = ttv.transforms.CenterCrop(50)
        result = crop(Tensor(img))
        assert result.shape == (3, 50, 50)


class TestPad:
    def test_uniform_padding(self):
        img = np.random.randn(10, 10).astype(np.float32)
        pad = ttv.transforms.Pad(2)
        result = pad(Tensor(img))
        assert result.shape == (14, 14)

    def test_with_fill_value(self):
        img = np.ones((5, 5), dtype=np.float32)
        pad = ttv.transforms.Pad(1, fill=0)
        result = pad(Tensor(img))
        result_np = result.numpy()
        # Borders should be 0
        assert result_np[0, 0] == 0.0
        assert result_np[0, -1] == 0.0
        # Center should be 1
        assert result_np[1, 1] == 1.0

    def test_asymmetric_padding(self):
        img = np.random.randn(10, 10).astype(np.float32)
        pad = ttv.transforms.Pad((1, 2, 3, 4))  # left, top, right, bottom
        result = pad(Tensor(img))
        assert result.shape == (16, 14)

    def test_3d_image(self):
        img = np.random.randn(10, 10, 3).astype(np.float32)
        pad = ttv.transforms.Pad(2)
        result = pad(Tensor(img))
        assert result.shape == (14, 14, 3)
