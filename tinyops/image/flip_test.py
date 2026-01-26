import cv2
import numpy as np
import pytest
from tinygrad import Tensor

from tinyops._core import assert_close, assert_one_kernel
from tinyops.image.flip import flip


def _get_input(shape):
    data = np.random.randn(*shape).astype(np.float32)
    return Tensor(data).realize(), data


@pytest.mark.parametrize("shape", [(10, 20), (10, 20, 3)])
@pytest.mark.parametrize("axis", [0, 1, -1])
@assert_one_kernel
def test_flip(shape, axis):
    tensor, data = _get_input(shape)

    result = flip(tensor, axis).realize()
    expected = cv2.flip(data, axis)
    assert_close(result, expected)
