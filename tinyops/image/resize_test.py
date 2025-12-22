import cv2
import numpy as np
import pytest
from tinygrad import Tensor
from tinyops.image.resize import resize, INTER_NEAREST, INTER_LINEAR
from tinyops._core import assert_close
from tinyops.test_utils import assert_one_kernel

def _get_input_2d():
    data = np.random.randn(10, 20).astype(np.float32)
    return Tensor(data).realize(), data

def _get_input_3d():
    data_3d = np.random.randn(10, 20, 3).astype(np.float32)
    return Tensor(data_3d).realize(), data_3d

TEST_PARAMS = [
    (INTER_NEAREST, cv2.INTER_NEAREST, (5, 10), False),
    (INTER_NEAREST, cv2.INTER_NEAREST, (15, 25), True),
    (INTER_LINEAR, cv2.INTER_LINEAR, (5, 10), False),
    (INTER_LINEAR, cv2.INTER_LINEAR, (15, 25), True),
]

@pytest.mark.parametrize("interp,cv2_interp,dsize,is_3d", TEST_PARAMS)
@assert_one_kernel
def test_resize(interp, cv2_interp, dsize, is_3d):
    if is_3d:
        tensor, data = _get_input_3d()
    else:
        tensor, data = _get_input_2d()

    result = resize(tensor, dsize, interpolation=interp).realize()
    expected = cv2.resize(data, (dsize[1], dsize[0]), interpolation=cv2_interp)
    assert_close(result, expected, atol=1e-5, rtol=1e-5)
