import cv2
import numpy as np
import pytest
from tinygrad import Tensor

from tinyops._core import assert_close, assert_one_kernel
from tinyops.image.rotate import (
    ROTATE_90_CLOCKWISE,
    ROTATE_90_COUNTERCLOCKWISE,
    ROTATE_180,
    RotateCode,
    rotate,
)


def _get_input(shape):
    data = np.random.randn(*shape).astype(np.float32)
    return Tensor(data).realize(), data


@pytest.mark.parametrize("shape", [(10, 20), (10, 20, 3)])
@pytest.mark.parametrize("rotate_code", [ROTATE_90_CLOCKWISE, ROTATE_180, ROTATE_90_COUNTERCLOCKWISE])
@assert_one_kernel
def test_rotate(shape, rotate_code):
    tensor, data = _get_input(shape)

    result = rotate(tensor, rotate_code).realize()

    cv2_code = {
        ROTATE_90_CLOCKWISE: cv2.ROTATE_90_CLOCKWISE,
        ROTATE_180: cv2.ROTATE_180,
        ROTATE_90_COUNTERCLOCKWISE: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }[rotate_code]

    expected = cv2.rotate(data, cv2_code)
    assert_close(result, expected)


@assert_one_kernel
def test_rotate_enum_usage():
    tensor, data = _get_input((10, 20))

    # Test using the Enum member directly
    result = rotate(tensor, RotateCode.CLOCKWISE_90).realize()
    expected = cv2.rotate(data, cv2.ROTATE_90_CLOCKWISE)

    assert_close(result, expected)
