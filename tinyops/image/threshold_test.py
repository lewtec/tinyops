import cv2
import numpy as np
import pytest
from tinygrad import Tensor

from tinyops._core import assert_close, assert_one_kernel
from tinyops.image.threshold import (
    THRESH_BINARY,
    THRESH_BINARY_INV,
    THRESH_TOZERO,
    THRESH_TOZERO_INV,
    THRESH_TRUNC,
    ThresholdType,
    threshold,
)


def _get_input():
    src = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    return Tensor(src.astype(np.float32)).realize(), src


THRESH_PARAMS = [
    (THRESH_BINARY, cv2.THRESH_BINARY),
    (THRESH_BINARY_INV, cv2.THRESH_BINARY_INV),
    (THRESH_TRUNC, cv2.THRESH_TRUNC),
    (THRESH_TOZERO, cv2.THRESH_TOZERO),
    (THRESH_TOZERO_INV, cv2.THRESH_TOZERO_INV),
]


@pytest.mark.parametrize("type, cv2_type", THRESH_PARAMS)
@assert_one_kernel
def test_threshold(type, cv2_type):
    src_tensor, src = _get_input()
    thresh = 127.0
    maxval = 255.0

    _, expected = cv2.threshold(src, thresh, maxval, cv2_type)
    result = threshold(src_tensor, thresh, maxval, type).realize()

    assert_close(result, Tensor(expected.astype(np.float32)))


@assert_one_kernel
def test_threshold_enum_usage():
    src_tensor, src = _get_input()
    thresh = 127.0
    maxval = 255.0

    # Test using the Enum member directly
    _, expected = cv2.threshold(src, thresh, maxval, cv2.THRESH_BINARY)
    result = threshold(src_tensor, thresh, maxval, ThresholdType.BINARY).realize()

    assert_close(result, Tensor(expected.astype(np.float32)))
